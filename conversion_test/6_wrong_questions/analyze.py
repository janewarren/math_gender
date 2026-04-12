#!/usr/bin/env python3
"""
Analyze iterative wrong-question results.

Reads the master.tsv (with first_correct_round annotations) and round files
to produce summary statistics and figures.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

try:
    import scienceplots
    plt.style.use(["science", "ieee", "no-latex"])
except ImportError:
    pass

sns.set_style("whitegrid", {"grid.alpha": 0.25, "grid.linewidth": 0.4})

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 9, "axes.linewidth": 0.6,
})

PAGE_W = 7.2
GOLDEN = 1.618

REPRO_DIR = Path(__file__).resolve().parent
ROUNDS_DIR = REPRO_DIR / "rounds"
FIGURES_DIR = REPRO_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COND_COLORS = {"math_only": "#332288", "no_guide": "#44AA99"}
COND_LABELS = {"math_only": "Math Only", "no_guide": "No Guide"}


def save_fig(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
    print(f"  → saved {name}.pdf / .png")


def main():
    master = pd.read_csv(REPRO_DIR / "master.tsv", sep="\t")
    total = len(master)
    never_correct = master[master["first_correct_round"] == -1]
    ever_correct = master[master["first_correct_round"] > 0]

    # ── 1. Top-level summary ─────────────────────────────────────
    max_round = int(ever_correct["first_correct_round"].max()) if len(ever_correct) else 0
    print("=" * 60)
    print("  ITERATIVE WRONG-QUESTION EXPERIMENT — RESULTS")
    print("=" * 60)
    print(f"\nTotal questions:       {total:,}")
    print(f"Eventually correct:    {len(ever_correct):,} ({len(ever_correct)/total*100:.1f}%)")
    print(f"Never correct:         {len(never_correct):,} ({len(never_correct)/total*100:.1f}%)")
    print(f"Rounds completed:      {max_round}")

    # ── 2. Survival curve: remaining wrong questions per round ───
    round_files = sorted(ROUNDS_DIR.glob("round_*.tsv"))
    survival = []
    for rf in round_files:
        rnum = int(rf.stem.split("_")[1])
        rdf = pd.read_csv(rf, sep="\t")
        n_sent = len(rdf)
        n_wrong = (~rdf["is_correct"]).sum()
        survival.append({"round": rnum, "sent": n_sent, "still_wrong": int(n_wrong),
                         "correct_this_round": int(rdf["is_correct"].sum())})
    surv_df = pd.DataFrame(survival)

    if len(surv_df):
        fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W / GOLDEN / 1.5))
        ax.plot(surv_df["round"], surv_df["still_wrong"], "o-", color="#332288",
                markersize=4, linewidth=1.2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Questions Still Wrong")
        ax.set_title("Survival Curve: Questions Remaining Wrong After Each Round")
        ax.grid(alpha=0.3)
        if surv_df["still_wrong"].max() > 1000:
            ax.set_yscale("log")
        plt.tight_layout()
        save_fig(fig, "survival_curve")
        plt.show()

    # ── 3. When are questions first answered correctly? ───────────
    if len(ever_correct):
        fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W / GOLDEN / 1.5))
        bins = range(1, max_round + 2)
        ax.hist(ever_correct["first_correct_round"], bins=bins, color="#44AA99",
                edgecolor="white", linewidth=0.4, align="left")
        ax.set_xlabel("Round First Answered Correctly")
        ax.set_ylabel("# Questions")
        ax.set_title("Distribution of First-Correct Round")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_fig(fig, "first_correct_distribution")
        plt.show()

    # ── 4. Never-correct by domain × condition ───────────────────
    print("\n" + "=" * 60)
    print("  NEVER-CORRECT QUESTIONS BY DOMAIN × CONDITION")
    print("=" * 60 + "\n")

    nc_cross = never_correct.groupby(["domain", "condition"]).size().unstack("condition", fill_value=0)
    total_cross = master.groupby(["domain", "condition"]).size().unstack("condition", fill_value=0)
    pct_cross = (nc_cross / total_cross * 100).round(1).fillna(0)

    print("Counts:")
    nc_cross["total"] = nc_cross.sum(axis=1)
    print(nc_cross.sort_values("total", ascending=False).to_string())
    print("\nAs % of questions in that cell:")
    print(pct_cross.to_string())

    # Bar chart
    if len(never_correct):
        nc_by_domain = never_correct.groupby("domain").size().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W / GOLDEN / 1.5))
        nc_by_domain.plot.barh(color="#CC6677", edgecolor="white", linewidth=0.4, ax=ax)
        ax.set_xlabel("# Never-Correct Questions")
        ax.set_title(f"Questions Never Answered Correctly ({len(never_correct):,} total)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_fig(fig, "never_correct_by_domain")
        plt.show()

    # ── 5. Difficulty breakdown of never-correct ──────────────────
    if "difficulty" in never_correct.columns:
        print("\n" + "=" * 60)
        print("  DIFFICULTY BREAKDOWN")
        print("=" * 60 + "\n")

        diff_nc = never_correct["difficulty"].value_counts()
        diff_all = master["difficulty"].value_counts()
        diff_compare = pd.DataFrame({"never_correct": diff_nc, "total": diff_all}).fillna(0).astype(int)
        diff_compare["pct_never_correct"] = (diff_compare["never_correct"] / diff_compare["total"] * 100).round(1)
        print(diff_compare.to_string())

    # ── 6. Sample never-correct questions ─────────────────────────
    print("\n" + "=" * 60)
    print("  SAMPLE NEVER-CORRECT QUESTIONS (up to 5 per domain)")
    print("=" * 60)

    for domain in sorted(never_correct["domain"].unique()):
        dom_qs = never_correct[never_correct["domain"] == domain]
        print(f"\n  {domain.upper()} ({len(dom_qs)} questions)")
        for _, row in dom_qs.head(5).iterrows():
            prompt_short = str(row["prompt"])[:150]
            print(f"    condition={row['condition']}  number={row['number']}  expected={row['answer']}")
            print(f"    prompt: {prompt_short}...")

    # ── 7. Save summary CSV ──────────────────────────────────────
    summary_path = REPRO_DIR / "summary.csv"
    summary = master[["row_id", "domain", "condition", "number", "answer",
                       "difficulty", "first_correct_round"]].copy()
    summary["never_correct"] = summary["first_correct_round"] == -1
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSaved summary to {summary_path}")

    if len(surv_df):
        print("\n" + "=" * 60)
        print("  ROUND-BY-ROUND SUMMARY")
        print("=" * 60 + "\n")
        print(surv_df.to_string(index=False))


if __name__ == "__main__":
    main()
