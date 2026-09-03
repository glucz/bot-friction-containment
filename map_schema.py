"""
Phase 1 -- map the live AGWA schema (run after db_config.ini is filled).

    python map_schema.py

Cheap metadata first (information_schema, DESCRIBE, SHOW INDEX -- all instant),
then a few BOUNDED probes that the session timeout will auto-abort if they try
to scan the whole 6-billion-row hits table. Nothing here writes. Output is
printed and saved to outputs/SCHEMA_MAP.md for us to confirm together.

The expected structure (from the data descriptor) is:
  hits(h_id, h_a_id->agent, h_ccode, h_mode, h_i_id->ip, h_d_id->domain,
       h_u_id->url, h_status, h_ts)         ~6.05e9 rows
  agent(a_id, a_name)                        ~2.27e6 rows
  domain(d_id, d_name)                       8780 rows
  url(u_id, u_name, u_last10)                ~7.12e7 rows
  ip(i_id, i_name, i_agent, i_year, i_month, i_count, i_ccode)  ~5.28e7 rows
"""
from __future__ import annotations

import config as pkg_config
import db

EXPECTED_TABLES = ["hits", "agent", "domain", "url", "ip"]
OUT = pkg_config.OUT_DIR / "SCHEMA_MAP.md"


def main() -> None:
    pkg_config.ensure_dirs()
    lines: list[str] = ["# AGWA live schema map\n"]

    info = db.ping()
    lines.append(f"- Server: `{info['version']}`  |  DB: `{info['db']}`  |  User: `{info['user']}`\n")
    print("Connected:", info)

    # --- table inventory (metadata only: instant) --------------------------
    inv = db.read_sql(
        """
        SELECT table_name, engine, table_rows, ROUND(data_length/1073741824,2) AS data_gb,
               ROUND(index_length/1073741824,2) AS index_gb
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
        ORDER BY table_rows DESC
        """
    )
    lines.append("## Tables (information_schema estimates)\n")
    lines.append(inv.to_markdown(index=False))
    lines.append("")
    print(inv.to_string(index=False))

    present = set(inv["table_name"].str.lower())

    # --- per-table columns + indexes ---------------------------------------
    for t in EXPECTED_TABLES:
        if t not in present:
            lines.append(f"## `{t}` -- NOT FOUND (table name may differ)\n")
            continue
        cols = db.read_sql(f"DESCRIBE `{t}`")
        idx = db.read_sql(f"SHOW INDEX FROM `{t}`")
        lines.append(f"## `{t}` columns\n")
        lines.append(cols.to_markdown(index=False))
        lines.append(f"\n### `{t}` indexes\n")
        idx_small = idx[["Key_name", "Seq_in_index", "Column_name", "Non_unique"]] if not idx.empty else idx
        lines.append(idx_small.to_markdown(index=False) if not idx.empty else "_(none)_")
        lines.append("")
        print(f"\n[{t}] columns: {list(cols['Field'])}")

    # --- bounded probes (timeout-guarded) ----------------------------------
    lines.append("## Probes\n")

    # date range -- MIN/MAX use the index if h_ts is indexed; else timeout aborts.
    _probe(lines, "hits date range (h_ts)",
           "SELECT MIN(h_ts) AS first_ts, MAX(h_ts) AS last_ts FROM hits")

    # status-code presence: is h_status indexed? if so this is cheap-ish.
    _probe(lines, "distinct HTTP status codes (confirm 403/429 exist)",
           "SELECT h_status, COUNT(*) AS n FROM hits GROUP BY h_status ORDER BY n DESC LIMIT 25")

    # robots.txt detection via url.u_last10 (last 10 chars of filename, cleartext)
    _probe(lines, "robots.txt url rows (u_last10 ending in robots.txt)",
           "SELECT u_id, u_last10 FROM url WHERE u_last10 LIKE '%robots.txt' LIMIT 5")

    # request-method presence (filled from 2021)
    _probe(lines, "request methods (h_mode)",
           "SELECT h_mode, COUNT(*) AS n FROM hits GROUP BY h_mode ORDER BY n DESC LIMIT 15")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSchema map written to {OUT}")


def _probe(lines: list[str], title: str, sql: str) -> None:
    lines.append(f"### {title}\n")
    print(f"\n--- probe: {title}")
    try:
        df = db.read_sql(sql)
        lines.append(df.to_markdown(index=False))
        print(df.to_string(index=False))
    except Exception as e:
        msg = f"_(probe failed or timed out: {type(e).__name__}: {str(e)[:120]})_"
        lines.append(msg)
        print(msg)
    lines.append("")


if __name__ == "__main__":
    main()
