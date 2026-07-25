"""
Read-only access layer for the AGWA MySQL database.

Safety is structural, not just intentional:
  - Credentials come from the gitignored db_config.ini (never from code/chat).
  - Every connection sets a session `max_execution_time` so any SELECT
    auto-aborts instead of scanning 6 billion rows forever.
  - `read_sql()` refuses anything that is not a read statement
    (SELECT / WITH / SHOW / DESCRIBE / EXPLAIN). Writes raise immediately.

Use a read-only DB grant as well, so the guarantee holds even if this code is
bypassed.
"""
from __future__ import annotations

import configparser
import re
from pathlib import Path

import pandas as pd

import config as pkg_config

CONFIG_PATH = pkg_config.HERE / "db_config.ini"

_READ_PREFIXES = ("select", "with", "show", "describe", "desc", "explain")
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|grant|revoke|"
    r"rename|lock|call|load|into\s+outfile|set\s+(?!session\s+max_execution_time))\b",
    re.IGNORECASE,
)


def _load_cfg() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Fill in {CONFIG_PATH} first (host/user/password/database).")
    cp = configparser.ConfigParser()
    cp.read(CONFIG_PATH)
    if "mysql" not in cp:
        raise ValueError("db_config.ini is missing a [mysql] section.")
    m = cp["mysql"]
    if not m.get("password") and not m.get("user"):
        raise ValueError("db_config.ini looks unfilled (no user/password).")
    return {
        "host": m.get("host"),
        "port": m.getint("port", 3306),
        "user": m.get("user"),
        "password": m.get("password", ""),
        "database": m.get("database"),
        "connect_timeout": m.getint("connect_timeout", 15),
        "max_execution_time_ms": m.getint("max_execution_time_ms", 60000),
        "use_ssl": m.getboolean("use_ssl", False),
        "ssl_ca": m.get("ssl_ca", "") or None,
    }


def get_connection(timeout_ms: int | None = None):
    """Open a pymysql connection with a session statement timeout.

    `timeout_ms` overrides the config default (e.g. extraction uses a tighter
    cap so giant-agent queries abort fast and the worker moves on)."""
    import pymysql

    cfg = _load_cfg()
    tmo = int(timeout_ms if timeout_ms is not None else cfg["max_execution_time_ms"])
    ssl = None
    if cfg["use_ssl"]:
        ssl = {"ca": cfg["ssl_ca"]} if cfg["ssl_ca"] else {}
    conn = pymysql.connect(
        host=cfg["host"], port=cfg["port"], user=cfg["user"],
        password=cfg["password"], database=cfg["database"],
        connect_timeout=cfg["connect_timeout"], ssl=ssl,
        cursorclass=pymysql.cursors.SSCursor,  # server-side cursor: stream big results
        autocommit=True,
    )
    with conn.cursor() as cur:
        # Auto-abort any runaway SELECT. MariaDB 10.4 here uses max_statement_time
        # (SECONDS); MySQL uses max_execution_time (ms). Set whichever applies.
        try:
            cur.execute("SET SESSION max_statement_time=%s", (tmo / 1000.0,))
        except Exception:
            try:
                cur.execute("SET SESSION max_execution_time=%s", (tmo,))
            except Exception:
                pass
    return conn


def _assert_read_only(sql: str) -> None:
    stripped = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)        # strip /* */ comments
    stripped = re.sub(r"--[^\n]*", " ", stripped)                     # strip -- comments
    stripped = stripped.strip().lower()
    if not stripped.startswith(_READ_PREFIXES):
        raise PermissionError(f"Refused: only read statements allowed. Got: {sql[:60]!r}")
    if _FORBIDDEN.search(stripped):
        raise PermissionError(f"Refused: write/DDL keyword detected in: {sql[:80]!r}")


def read_sql(sql: str, params: tuple | dict | None = None, conn=None) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame. Opens/closes its own
    connection unless one is passed in."""
    _assert_read_only(sql)
    own = conn is None
    if own:
        conn = get_connection()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        if own:
            conn.close()


def ping() -> dict:
    """Connectivity check: returns server version + current database."""
    df = read_sql("SELECT VERSION() AS version, DATABASE() AS db, CURRENT_USER() AS user")
    return df.iloc[0].to_dict()
