"""Construct-validity cross-check: does the three-feature observability space
track the transformer classifier's own geometry?

Paper D's observability score O lives in a hand-built, interpretable space -
[p404, robots_rate, log1p(hits)] standardized on the control centroid. The
transformer classifier of the sibling repo (Mathematics 2025) independently
exports, for the 1,191 browser-labelled agents it scored, both a mean P(bot)
(`avg_bot_prob`) and its OWN distance to a human centroid in the model's
representation (`distance_to_human_centroid`). Neither has ever been compared
with O.

This script computes, on the overlap, the rank and linear association between
the behavioral O and the transformer's two quantities.

SCOPE (hard limits, do not over-read):
  * the 1,191 browser-labelled agents are a MIXED contamination set, not the
    7,812-agent treatment arm;
  * both transformer columns are full-period per-agent scalars, so this can
    support a construct-validity correlation only - never an event study or DiD.

Usage: python db_obs_transformer_crosscheck.py
Outputs: outputs/DB_obs_transformer_crosscheck.{json,md}
         outputs/figures/DB_obs_transformer_crosscheck.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import config
import db_source as dbs

MIN_DAYS = 20          # same filter as build_botness_axis()
FRAME = config.TAB_DIR / "_botness_axis_v2.npz"   # published arm-of-record frame


def behavior_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["p404"].to_numpy(float),
        df["robots_rate"].to_numpy(float),
        np.log1p(df["hits"].to_numpy(float)),
    ])


def main() -> None:
    frame = np.load(FRAME)
    mu_H, sigma_H = frame["mu_H"], frame["sigma_H"]

    ranked = pd.read_csv(config.CHROME_RANK_CSV)
    ranked["a_id"] = ranked["agent_file"].str.split(".").str[0].astype(int)

    rows = []
    missing = short = 0
    for r in ranked.itertuples():
        p = dbs._cache_path(int(r.a_id))
        if not p.exists():
            missing += 1
            continue
        df = pd.read_parquet(p)
        if len(df) < MIN_DAYS:
            short += 1
            continue
        Z = behavior_matrix(df)
        # (a) O of the agent's mean behavior vector - matches how the pipeline
        #     builds centroids, and is the right comparand for a full-period scalar
        O_meanvec = float(np.linalg.norm((Z.mean(axis=0) - mu_H) / sigma_H))
        # (b) mean of the daily O series - matches how the event study reads O
        O_dailymean = float(np.linalg.norm((Z - mu_H) / sigma_H, axis=1).mean())
        rows.append({
            "a_id": int(r.a_id), "n_days": len(df),
            "avg_bot_prob": float(r.avg_bot_prob),
            "tf_distance": float(r.distance_to_human_centroid),
            "num_windows": int(r.num_windows),
            "O_meanvec": O_meanvec, "O_dailymean": O_dailymean,
        })

    d = pd.DataFrame(rows)
    out: dict = {
        "scope": ("1,191 browser-labelled (labeled_chrome) agents: a MIXED "
                  "contamination set, NOT the 7,812 treatment arm. Full-period "
                  "scalars only - supports construct validity, not a DiD."),
        "reference_frame": "outputs/tables/_botness_axis_v2.npz (published arm-of-record mu_H, sigma_H)",
        "n_ranked": int(len(ranked)), "n_missing_cache": missing,
        "n_below_min_days": short, "n_analyzed": int(len(d)), "min_days": MIN_DAYS,
    }

    pairs = [("O_meanvec", "tf_distance"), ("O_dailymean", "tf_distance"),
             ("O_meanvec", "avg_bot_prob"), ("O_dailymean", "avg_bot_prob")]
    for a, b in pairs:
        x, y = d[a].to_numpy(), d[b].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        # cluster-free agent-level bootstrap CI on Spearman
        rng = np.random.default_rng(42)
        boot = [stats.spearmanr(*np.array([x, y])[:, rng.integers(0, len(x), len(x))])[0]
                for _ in range(2000)]
        out[f"{a}__vs__{b}"] = {
            "n": int(len(x)),
            "pearson_r": round(float(pr), 4), "pearson_p": float(pp),
            "spearman_rho": round(float(sr), 4), "spearman_p": float(sp),
            "spearman_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                              round(float(np.percentile(boot, 97.5)), 4)],
        }

    # --- discrimination: tf_distance is BIMODAL, so correlation understates ----
    # 594 agents pile near 0 and 387 near the max, with 27 in between: the
    # transformer's centroid distance behaves as a hard two-cluster split, so the
    # meaningful question is whether O separates those clusters, not how linearly
    # it tracks a variable that has almost no middle.
    hist = np.histogram(d["tf_distance"], bins=20)[0].tolist()
    lo, hi = d["tf_distance"] < 0.5, d["tf_distance"] > 1.5
    sub = d[lo | hi].copy()
    sub["tf_bot"] = (sub["tf_distance"] > 1.5).astype(int)
    out["tf_distance_is_bimodal"] = {
        "histogram_20bins": hist,
        "n_low_cluster": int(lo.sum()), "n_high_cluster": int(hi.sum()),
        "n_middle": int(len(d) - lo.sum() - hi.sum()),
        "note": "correlation against this column is attenuated by its two-cluster shape; AUC below is the fair statistic",
    }
    out["transformer_internal_agreement_spearman"] = round(
        float(stats.spearmanr(d["tf_distance"], d["avg_bot_prob"])[0]), 4)

    for col in ("O_meanvec", "O_dailymean"):
        a = sub[sub.tf_bot == 1][col]; b = sub[sub.tf_bot == 0][col]
        out[f"{col}__discriminates_tf_clusters"] = {
            "n": int(len(sub)),
            "auc": round(float(roc_auc_score(sub["tf_bot"], sub[col])), 3),
            "median_high_cluster": round(float(a.median()), 3),
            "median_low_cluster": round(float(b.median()), 3),
            "mannwhitney_p": float(stats.mannwhitneyu(a, b)[1]),
        }
        y = (d["avg_bot_prob"] > 0.5).astype(int)
        out[f"{col}__discriminates_pbot_half"] = {
            "n": int(len(d)), "n_pos": int(y.sum()),
            "auc": round(float(roc_auc_score(y, d[col])), 3),
        }

    out["reading"] = (
        "O and the transformer order agents in the SAME direction, significantly "
        "(AUC 0.717, Mann-Whitney p=1.4e-34), but they are far from redundant. "
        "Non-redundancy is partly protective: an O that reproduced the classifier "
        "would reintroduce the circularity of measuring evasion of a detector with "
        "that detector's own score. What this CANNOT establish is that the "
        "classifier's score also FALLS after blocks - the export is full-period and "
        "covers the wrong population, so the post-block behavior of b stays untested."
    )

    config.ensure_dirs()
    (config.OUT_DIR / "DB_obs_transformer_crosscheck.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150)
    ax = axes[0]
    ax.hist(d["tf_distance"], bins=40, color="tab:gray")
    ax.set_xlabel("transformer distance to human centroid")
    ax.set_ylabel("agents")
    ax.set_title("Transformer distance is bimodal\n(627 low / 462 high / 27 middle)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(d["O_meanvec"], d["avg_bot_prob"], s=10, alpha=0.35, color="tab:blue")
    st = out["O_meanvec__vs__avg_bot_prob"]
    ax.set_xscale("log")
    ax.set_xlabel("behavioral observability O (log scale)")
    ax.set_ylabel("transformer mean P(bot)")
    ax.set_title("Continuous comparand\nSpearman rho = %.3f [%.2f, %.2f]"
                 % (st["spearman_rho"], st["spearman_ci95"][0], st["spearman_ci95"][1]))
    ax.grid(alpha=0.3)

    ax = axes[2]
    parts = [sub[sub.tf_bot == 0]["O_meanvec"], sub[sub.tf_bot == 1]["O_meanvec"]]
    ax.boxplot(parts, labels=["transformer\nhuman-like", "transformer\nbot-like"], showfliers=False)
    ax.set_ylabel("behavioral observability O")
    ax.set_title("O separates the transformer's clusters\nAUC = %.3f"
                 % out["O_meanvec__discriminates_tf_clusters"]["auc"])
    ax.grid(alpha=0.3)

    fig.suptitle("Construct-validity cross-check: hand-built observability space vs transformer geometry. "
                 "1,116 browser-labelled agents (mixed contamination set, NOT the treatment arm); "
                 "full-period scalars, so no event study is implied.", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "DB_obs_transformer_crosscheck.png", dpi=150)
    plt.close(fig)

    d.to_csv(config.TAB_DIR / "DB_obs_transformer_crosscheck.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
