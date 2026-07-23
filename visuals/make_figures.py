"""FIGURES for the thesis — reads out/*.csv (read-only), writes visuals/figures/*.

This folder is fully separate from pipeline/: it never imports pipeline code
and never writes anywhere under out/. Deleting visuals/ has zero effect on
the pipeline or its tests.

Setup (one-time):
    python3 -m venv visuals/.venv
    visuals/.venv/bin/pip install -r visuals/requirements.txt

Run:
    visuals/.venv/bin/python visuals/make_figures.py

Each figure is saved as both .png (300dpi, for drafts/slides) and .pdf
(vector, for the LaTeX/Word thesis document) in visuals/figures/.

Color use follows a fixed categorical order so the same entity (e.g.
"build-leaning") always gets the same color across every figure in the
thesis: slot 1 blue = build, slot 2 green = buy, slot 3 yellow = mixed,
gray = unclear/too_short. Single-series ranked bars use one flat hue
(sequential blue) since the bar length already encodes magnitude.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "out")
FIG_DIR = os.path.join(HERE, "figures")

# ---- palette (validated categorical order — see dataviz skill palette.md) ----
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"      # slot 1 — build-leaning
GREEN = "#008300"     # slot 2 — buy-leaning
YELLOW = "#eda100"    # slot 3 — mixed
GRAY_UNCLEAR = "#c3c2b7"
GRAY_SHORT = "#e1e0d9"

STANCE_COLORS = {
    "build-leaning": BLUE,
    "buy-leaning": GREEN,
    "mixed": YELLOW,
    "unclear": GRAY_UNCLEAR,
    "too_short": GRAY_SHORT,
}
STANCE_ORDER = ["build-leaning", "buy-leaning", "mixed", "unclear", "too_short"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.titlecolor": INK_PRIMARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _clean_axes(ax, hide_spines=("top", "right")):
    for s in hide_spines:
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(BASELINE)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1 — corpus composition by subreddit
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# ---------------------------------------------------------------------------
def fig_corpus_by_subreddit(master):
    counts = master["subreddit"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(counts.index, counts.values, color=BLUE, height=0.65)
    for bar, val in zip(bars, counts.values):
        ax.text(val + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Threads kept")
    ax.set_title("Corpus composition by subreddit", loc="left", fontsize=11, pad=12)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    fig.text(0.01, -0.02,
              "Note: r/Lawyertalk and r/legaltech together account for the large majority "
              "of the corpus; the remaining 10 subreddits contribute a small share each.",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig1_corpus_by_subreddit")


# ---------------------------------------------------------------------------
# Figure 2 — corpus growth over time
# Job: trend over time -> line, single series, sequential hue.
# ---------------------------------------------------------------------------
def fig_corpus_over_time(master):
    dates = pd.to_datetime(master["created_utc"], errors="coerce")
    per_year = dates.dt.year.value_counts().sort_index()
    per_year = per_year[per_year.index >= 2015]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(per_year.index, per_year.values, color=BLUE, linewidth=2, marker="o",
            markersize=4, markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(per_year.index, per_year.values, color=BLUE, alpha=0.08)
    ax.set_ylabel("Threads kept")
    ax.set_title("Corpus growth over time", loc="left", fontsize=11, pad=12)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _clean_axes(ax)
    fig.text(0.01, -0.02,
              "Note: partial-year data at both ends of the range depending on dump coverage.",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig2_corpus_over_time")


# ---------------------------------------------------------------------------
# Figure 3 — factor salience ranking
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# ---------------------------------------------------------------------------
def fig_factor_salience(factors, top_n=12):
    df = factors.sort_values("pct_decision_threads", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.barh(df["label"], df["pct_decision_threads"], color=BLUE, height=0.65)
    for bar, val in zip(bars, df["pct_decision_threads"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%",
                 va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Share of decision-tier threads mentioning the factor (%)")
    ax.set_title(f"Top {top_n} factors by decision-tier salience", loc="left",
                 fontsize=11, pad=12)
    ax.set_xlim(0, max(df["pct_decision_threads"]) * 1.15)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    _save(fig, "fig3_factor_salience")


# ---------------------------------------------------------------------------
# Figure 4 — provider mentions ranking
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# Category shown as a muted text annotation instead of color (14 categories
# is far past the categorical series cap of ~8).
# ---------------------------------------------------------------------------
def fig_provider_mentions(providers, top_n=15):
    df = providers.sort_values("n_threads", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(df["provider"], df["n_threads"], color=BLUE, height=0.65)
    xmax = max(df["n_threads"]) * 1.25
    for bar, val, cat in zip(bars, df["n_threads"], df["category"]):
        ax.text(val + xmax * 0.01, bar.get_y() + bar.get_height() / 2, f"{val:,}",
                 va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
        ax.text(xmax * 0.985, bar.get_y() + bar.get_height() / 2, cat,
                 va="center", ha="right", fontsize=7.5, color=INK_MUTED, style="italic")
    ax.set_xlabel("Threads mentioning the provider")
    ax.set_title(f"Top {top_n} providers by thread coverage", loc="left",
                 fontsize=11, pad=12)
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    _save(fig, "fig4_provider_mentions")


# ---------------------------------------------------------------------------
# Figure 5 — stance distribution by corpus tier
# Job: tell distinct series apart -> grouped bar, categorical (2 series:
# decision-tier vs discourse-tier), fixed stance-color order on the x-axis.
# ---------------------------------------------------------------------------
def fig_stance_by_tier(master):
    counts = (master.groupby(["stance_heuristic", "corpus_tier"])
              .size().unstack(fill_value=0))
    counts = counts.reindex(STANCE_ORDER)
    tier1 = counts.get(1, pd.Series(0, index=STANCE_ORDER))
    tier2 = counts.get(2, pd.Series(0, index=STANCE_ORDER))

    x = range(len(STANCE_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([i - width / 2 for i in x], tier1.values, width, label="Decision-tier",
           color=BLUE)
    ax.bar([i + width / 2 for i in x], tier2.values, width, label="Discourse-tier",
           color=INK_MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("-", "-\n") if s == "build-leaning" or s == "buy-leaning"
                         else s for s in STANCE_ORDER], fontsize=9)
    ax.set_ylabel("Threads")
    ax.set_title("Stance classification by corpus tier", loc="left", fontsize=11, pad=12)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _clean_axes(ax)
    fig.text(0.01, -0.03,
              "'too_short' = fewer than 3 scraped comments; excluded from stance analysis "
              "by design, not genuinely ambiguous.",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig5_stance_by_tier")


# ---------------------------------------------------------------------------
# Figure 6 — decision-tone sentiment by stance
# Job: compare distributions across a few categories -> boxplot, categorical
# color matching the same build/buy hues used in Figure 5.
# ---------------------------------------------------------------------------
def fig_sentiment_by_stance(master):
    df = master[master["n_decision_sentences"].fillna(0).astype(float) > 0].copy()
    df = df[df["stance_heuristic"].isin(["build-leaning", "buy-leaning", "mixed"])]
    groups = [df.loc[df["stance_heuristic"] == s, "decision_tone_mean"].dropna()
              for s in ["build-leaning", "buy-leaning", "mixed"]]
    labels = [f"Build-leaning\n(n={len(g)})" for g in groups[:1]] + \
              [f"Buy-leaning\n(n={len(groups[1])})"] + \
              [f"Mixed\n(n={len(groups[2])})"]
    colors = [BLUE, GREEN, YELLOW]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5,
                     medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
                     whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED},
                     flierprops={"markeredgecolor": INK_MUTED, "markersize": 4})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    ax.axhline(0, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_ylabel("Decision-sentence tone (VADER, -1 to 1)")
    ax.set_title("Decision-tone sentiment by stance", loc="left", fontsize=11, pad=12)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    fig.text(0.01, -0.03,
              "Only tier-1 threads with >=1 qualifying decision sentence; a minority of "
              "the corpus (see run_summary.txt) — read as exploratory, not conclusive.",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig6_sentiment_by_stance")


def main():
    master = pd.read_csv(os.path.join(OUT_DIR, "threads_master.csv"))
    factors = pd.read_csv(os.path.join(OUT_DIR, "factor_salience.csv"))
    providers = pd.read_csv(os.path.join(OUT_DIR, "provider_mentions.csv"))

    print("Generating figures from out/ (read-only) ...")
    fig_corpus_by_subreddit(master)
    fig_corpus_over_time(master)
    fig_factor_salience(factors)
    fig_provider_mentions(providers)
    fig_stance_by_tier(master)
    fig_sentiment_by_stance(master)
    print(f"Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
