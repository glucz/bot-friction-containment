"""
Build the agent roster for the overnight battery and save it to outputs/roster.csv.

Roles (all filtered to a_id <= 2,147,483,647 so hits exist):
  bot_hp        honeypot-CONFIRMED bots: SUM(uh_total) > 0 in url2agent_honeypot
  ctrl_hpneg    honeypot-NEGATIVE controls: present in url2agent_honeypot but
                never hit a honeypot (SUM(uh_total) = 0). Likely-benign, NOT
                proven human -- labelled honestly.
  ctrl_white    seen from a whitelisted (operator-trusted) IP -> strong control
  labeled_*     the original local .bot/.human/.chrome/.pothuman agents, for
                direct comparability with the prior (local-file) analyses

Dedup priority when an agent qualifies for several: labeled_* > bot_hp >
ctrl_white > ctrl_hpneg. All source tables are small (<=194k / 279), so the
scans here are cheap; only the giant TokuDB tables are off-limits to scans.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

import config
import db
import db_source as dbs

ROSTER_CSV = config.OUT_DIR / "roster.csv"
HPNEG_SAMPLE = 6000    # honeypot-negative controls to draw
WHITE_SAMPLE = 5000    # cap whitelist controls (else ~17k inflates runtime)


def _honeypot_split() -> tuple[list[int], list[int]]:
    """(honeypot-positive bots, honeypot-negative agents), a_id<=MAX."""
    df = db.read_sql(
        """
        SELECT uh_a_id AS a_id, SUM(uh_total) AS tot
        FROM url2agent_honeypot
        WHERE uh_a_id <= %s
        GROUP BY uh_a_id
        """,
        (dbs.MAX_A_ID,),
    )
    pos = df.loc[df["tot"] > 0, "a_id"].astype(int).tolist()
    neg = df.loc[df["tot"] == 0, "a_id"].astype(int).tolist()
    return pos, neg


def _whitelist_agents() -> list[int]:
    """Agents ever seen from a whitelisted IP (i_name indexed -> cheap)."""
    try:
        df = db.read_sql(
            "SELECT DISTINCT i_agent AS a_id FROM ip "
            "WHERE i_name IN (SELECT w_name FROM whiteip) AND i_agent <= %s",
            (dbs.MAX_A_ID,),
        )
        return df["a_id"].dropna().astype(int).tolist()
    except Exception as e:
        print("  whitelist query failed (skipping):", str(e)[:90])
        return []


def _local_labeled() -> dict[int, str]:
    """Original local files -> {a_id: 'labeled_<label>'}."""
    out = {}
    for label, pattern in config.CLASSES.items():
        for fp in glob.glob(str(config.DATA_DIR / pattern)):
            stem = os.path.splitext(os.path.basename(fp))[0]
            try:
                aid = int(stem)
            except ValueError:
                continue
            if aid <= dbs.MAX_A_ID:
                out[aid] = f"labeled_{label}"
    return out


def build(seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    config.ensure_dirs()
    print("Querying honeypot split...")
    pos, neg = _honeypot_split()
    print(f"  honeypot-positive bots: {len(pos)}   honeypot-negative: {len(neg)}")
    white = _whitelist_agents()
    print(f"  whitelist agents: {len(white)} (capping to {WHITE_SAMPLE})")
    white_s = pd.Series(white)
    if len(white_s) > WHITE_SAMPLE:
        white = white_s.sample(WHITE_SAMPLE, random_state=seed).astype(int).tolist()
    labeled = _local_labeled()
    print(f"  local labeled agents: {len(labeled)}")

    # sample honeypot-negative controls
    neg_s = pd.Series(neg)
    if len(neg_s) > HPNEG_SAMPLE:
        neg_s = neg_s.sample(HPNEG_SAMPLE, random_state=seed)
    neg_sample = neg_s.astype(int).tolist()

    # assemble with dedup priority: labeled > bot_hp > ctrl_white > ctrl_hpneg
    role: dict[int, str] = {}
    for aid in neg_sample:
        role[aid] = "ctrl_hpneg"
    for aid in white:
        role[aid] = "ctrl_white"
    for aid in pos:
        role[aid] = "bot_hp"
    for aid, lab in labeled.items():
        role[aid] = lab  # labeled_* wins

    df = pd.DataFrame({"a_id": list(role), "role": list(role.values())})
    df["is_bot"] = df["role"].isin(["bot_hp"]) | df["role"].eq("labeled_bot")
    df["is_control"] = df["role"].str.startswith("ctrl_") | df["role"].isin(
        ["labeled_human", "labeled_chrome", "labeled_pothuman"])
    df = df.sort_values("a_id").reset_index(drop=True)
    df.to_csv(ROSTER_CSV, index=False)

    print("\nRoster composition:")
    print(df["role"].value_counts().to_string())
    print(f"\nTotal: {len(df)} agents  ->  {ROSTER_CSV}")
    return df


if __name__ == "__main__":
    build()
