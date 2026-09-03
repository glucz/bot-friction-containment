# AGWA DB mapping conclusions (2026-06-09)

Server: MariaDB 10.4.33, database `agentinfo`. Date span **2019-03-10 → 2023-03-05**, 2,274,125 agents, 6.05 B hits.

## Corrections to the published data descriptor
- `url` has **no `u_last10`**; columns are `u_name, u_extension, u_honeypot`. **`u_name` is CLEARTEXT, not hashed** (e.g. `robots.txt`, `index.php`, `login.php`). Huge: paths are directly queryable.
- `robots.txt` = **`u_id = 64`** -> RFC-9309 compliance per agent/date via `h_u_id=64` (indexes `h_2`, `h_6`).
- Precomputed `agent.a_stat_*` method columns are **0% populated**; `a_totalhit` ~19%, `a_date_*` trajectory ~15%. Do NOT rely on them; compute behavioral features from `hits` per agent.

## What's queryable cheaply vs. not
- **CHEAP (indexed):** per-agent slices `WHERE h_a_id=X` (idx `h_1`=a_id,ts and `h_5`=a_id,status); per-url slices `WHERE h_u_id=X` (idx `h_2`,`h_6`). Warm-cache ~0.3 s for a small agent; first hit is cold (~11 s) then fast.
- **FAILS:** global aggregates / unindexed full scans on the big TokuDB tables (`hits`, `url`) -> TokuDB "error 1152" or statement-timeout. Never `GROUP BY h_status` over all of `hits`, etc.

## Real HTTP status palette (confirmed on a honeypot URL)
200, 302, 404, 301, **401**, 500, 400, **403**, 508, 406, 422, 503, **421**, 303, 410, 405, 308, 307, 206, 103. The block/forbidden codes (401/403) absent from the derived files are present here.

## Labels available (ground truth)
- **Honeypot-hit bots:** `url2agent_honeypot` (uh_a_id, uh_u_id, uh_total). **38,355 agents** have uh_total>0 = high-confidence bots. (e.g. agent 445015 hit a honeypot 17.3 M times.)
- **Known-bad IPs:** `hackip` (2.97 M; h_name=IP, h_cnt, h_visits, h_date).
- **Whitelist:** `whiteip` (279; operator's own infra -> EXCLUDE).
- **Self-identification:** `agent.a_name` is the full cleartext UA string (e.g. `Mail.RU_Bot/Fast/2.0`, `Chrome/84...`, `Microsoft Outlook 16.0`).

## Filename -> agent mapping (validated)
Local `<id>.<label>` files map exactly to `agent.a_id`. Cross-check is illuminating:
- `1006025140.bot` = `Mail.RU_Bot/Fast/2.0` (self-IDed bot, hit 2 honeypots). ✓
- `1002085156.chrome` = `Chrome/84...` BUT hit **9 honeypots 5086×** -> a disguised bot wearing a Chrome UA (exactly the hidden-bot case the classifier targets). ✓
- `1032111909.human` = `Microsoft Outlook` client, no honeypot hits. ✓

## Enriched daily-series extraction query (the DBSource workhorse)
```sql
SELECT DATE(h_ts) AS day, COUNT(*) AS hits,
       SUM(h_status=200) c200, SUM(h_status=404) c404,
       SUM(h_status IN (401,403)) cblock, SUM(h_status=429) c429,
       SUM(h_u_id=64) robots,                          -- RFC-9309 compliance
       COUNT(DISTINCT h_i_id) n_ip, COUNT(DISTINCT h_d_id) n_dom,
       COUNT(DISTINCT h_ccode) n_ctry, COUNT(DISTINCT h_u_id) n_url
FROM hits WHERE h_a_id = %s GROUP BY DATE(h_ts) ORDER BY day;
```
Gives the **calendar-dated** daily vector with **real block codes** and **robots.txt access** — everything the derived files lacked.

## New capabilities this unlocks (vs. the local derived files)
1. Real calendar dates -> population co-evolution over 2019-2023 + dated before/after a block.
2. Real 401/403/429 block codes -> precise filter-pressure events (not the 1-p200 proxy).
3. robots.txt access log -> per-agent RFC-9309 compliance trajectories.
4. Ground-truth honeypot/hackip labels -> far stronger bot/human separation than self-ID alone, and 38k+ labeled bots (vs 177 local .bot files).
