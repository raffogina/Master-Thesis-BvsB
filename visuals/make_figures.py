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
import pandas as pd
from matplotlib.patches import Patch

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

# ---- static reference counts, sourced from the Arctic Shift local dumps ----
# (data/*.jsonl — total ~210MB across the 12 corpus subreddits). These are
# ALL submissions ever captured in the dump for that subreddit, i.e. the
# denominator for "what share of a subreddit's threads matched our query" —
# not filtered by query or tier. The dumps are large and static (re-scanning
# them on every figure run buys nothing), so the counts are captured here.
# Recompute (one-time line count / created_utc scan of data/r_<sub>_posts.jsonl)
# if data/ is refreshed with newer dumps.
SUBREDDIT_TOTAL_POSTS = {
    "Lawyertalk": 55460,
    "legaltech": 4956,
    "legaltechAI": 172,
    "LegaltechEurope": 49,
    "LegalAITech": 34,
    "AIforLawyers_": 30,
    "LegalAIHelp": 60,
    "TheAttorneyLounge": 13,
    "techlaw": 94,
    "LegalAIPrompts": 5,
    "LegalTechMakers": 42,
    "lawtech": 150,
}

# Total submissions posted per calendar year, summed across the same 12
# subreddits (same source as SUBREDDIT_TOTAL_POSTS). 2026 is partial: the
# dumps cover through 2026-06-29.
SUBREDDIT_TOTAL_POSTS_BY_YEAR = {
    2022: 6155,
    2023: 9751,
    2024: 14930,
    2025: 19655,
    2026: 10385,   # partial year, through 2026-06-29
}

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
# Figure 1 — query yield by subreddit
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# Metric: threads kept (post query + tier filter) as a % of ALL threads ever
# posted in that subreddit — not a raw count, so a small subreddit that is
# heavily build-vs-buy focused isn't visually dwarfed by a large one that
# mostly isn't. Subreddits below 1% are folded into a single "Other" bar.
# ---------------------------------------------------------------------------
def fig_corpus_by_subreddit(master):
    kept = master["subreddit"].value_counts()
    totals = pd.Series(SUBREDDIT_TOTAL_POSTS)
    df = pd.DataFrame({"kept": kept, "total": totals.reindex(kept.index)})
    df["pct"] = df["kept"] / df["total"] * 100

    small = df[df["pct"] < 1.0]
    large = df[df["pct"] >= 1.0].copy()
    if len(small):
        other = pd.DataFrame(
            {"kept": [small["kept"].sum()], "total": [small["total"].sum()]},
            index=[f"Other ({len(small)} subreddits)"])
        other["pct"] = other["kept"] / other["total"] * 100
        large = pd.concat([large, other])
    large = large.sort_values("pct")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(large.index, large["pct"], color=BLUE, height=0.65)
    for bar, val in zip(bars, large["pct"]):
        ax.text(val + max(large["pct"]) * 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Threads kept, as % of that subreddit's total threads")
    ax.set_title("Query yield by subreddit", loc="left", fontsize=11, pad=12)
    ax.set_xlim(0, max(large["pct"]) * 1.15)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    fig.text(0.01, -0.02,
              "Denominator = all submissions ever posted in that subreddit (Arctic Shift "
              "dump), not just query matches. Subreddits below 1% are grouped into 'Other'.",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig1_corpus_by_subreddit")


# ---------------------------------------------------------------------------
# Figure 2 — build vs. buy discussion presence
# Job: tell distinct series apart, over time -> grouped bar, two series
# (threads kept vs. total threads posted that year, all 12 subreddits).
# Years fixed at 2022-2026 (full data coverage). 2026 is partial (dumps run
# through 2026-06-29): the observed bar is solid, and a hatched segment on
# top projects the rest of the year at the same daily pace, so the bar isn't
# misread as the topic declining when it's really just an incomplete year.
# Log y-axis: "kept" and "total" differ by ~1-2 orders of magnitude, so a
# linear axis would flatten the kept series to invisibility.
# ---------------------------------------------------------------------------
def fig_build_vs_buy_presence(master):
    years = list(range(2022, 2027))
    dates = pd.to_datetime(master["created_utc"], errors="coerce", utc=True)
    kept_by_year = dates.dt.year.value_counts()
    kept_actual = {y: int(kept_by_year.get(y, 0)) for y in years}
    total_actual = {y: SUBREDDIT_TOTAL_POSTS_BY_YEAR[y] for y in years}

    cutoff = dates.max()
    current_year = int(cutoff.year)
    days_in_year = 366 if pd.Timestamp(current_year, 12, 31).is_leap_year else 365
    frac_elapsed = cutoff.dayofyear / days_in_year

    kept_proj = dict(kept_actual)
    total_proj = dict(total_actual)
    kept_proj[current_year] = kept_actual[current_year] / frac_elapsed
    total_proj[current_year] = total_actual[current_year] / frac_elapsed

    kept_solid = [kept_actual[y] for y in years]
    total_solid = [total_actual[y] for y in years]
    kept_remainder = [kept_proj[y] - kept_actual[y] for y in years]
    total_remainder = [total_proj[y] - total_actual[y] for y in years]

    x = list(range(len(years)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_yscale("log")

    xk = [i - width / 2 for i in x]
    xt = [i + width / 2 for i in x]
    ax.bar(xk, kept_solid, width, color=BLUE, zorder=3)
    ax.bar(xk, kept_remainder, width, bottom=kept_solid, color=BLUE, alpha=0.35,
           hatch="//", edgecolor="white", linewidth=0, zorder=3)
    ax.bar(xt, total_solid, width, color=INK_MUTED, zorder=3)
    ax.bar(xt, total_remainder, width, bottom=total_solid, color=INK_MUTED, alpha=0.35,
           hatch="//", edgecolor="white", linewidth=0, zorder=3)

    for xi, y in zip(xk, years):
        top = kept_proj[y]
        label = f"{kept_actual[y]:,}" if y != current_year else f"{kept_actual[y]:,}→{round(top):,}"
        ax.text(xi, top * 1.08, label, ha="center", va="bottom", fontsize=7.5,
                 color=INK_SECONDARY)
    for xi, y in zip(xt, years):
        top = total_proj[y]
        label = f"{total_actual[y]:,}" if y != current_year else f"{total_actual[y]:,}→{round(top):,}"
        ax.text(xi, top * 1.08, label, ha="center", va="bottom", fontsize=7.5,
                 color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(50, max(total_proj.values()) * 2.2)
    ax.set_ylabel("Threads (log scale)")
    ax.set_title("Build vs. buy discussion presence", loc="left", fontsize=11, pad=12)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    handles = [
        Patch(color=BLUE, label="Threads kept"),
        Patch(color=INK_MUTED, label="Total threads posted"),
        Patch(facecolor="white", edgecolor=INK_MUTED, hatch="//",
              label=f"Projected remainder of {current_year}"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8.5)
    _clean_axes(ax)
    fig.text(0.01, -0.03,
              f"{current_year} data covers Jan 1-{cutoff.strftime('%b %d')} "
              f"({cutoff.dayofyear} of {days_in_year} days); the hatched segment projects "
              "the rest of the year at the same daily pace. 'Total threads posted' = all "
              "submissions across the 12 corpus subreddits that year, kept or not.",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig2_build_vs_buy_presence")


# ---------------------------------------------------------------------------
# Figure 3 — factor salience ranking
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# Metric: share of ALL threads mentioning the factor (pct_all_threads), not
# just decision-tier threads. Axis fixed at 0-100% (the true scale of a
# percentage) rather than 1.15x the max bar, so bar length isn't inflated
# relative to the full possible range.
# ---------------------------------------------------------------------------
def fig_factor_salience(factors, top_n=12):
    df = factors.sort_values("pct_all_threads", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.barh(df["label"], df["pct_all_threads"], color=BLUE, height=0.65)
    for bar, val in zip(bars, df["pct_all_threads"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%",
                 va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Share of all threads mentioning the factor (%)")
    ax.set_title(f"Top {top_n} factors by thread salience", loc="left",
                 fontsize=11, pad=12)
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    _save(fig, "fig3_factor_salience")


# ---------------------------------------------------------------------------
# Figure 4 — provider mentions ranking
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# No category labels: with 14 categories the per-bar tags read as clutter
# rather than signal (see fig4_provider_mentions v1) — category-level
# comparison isn't the job of this figure, so it was dropped rather than
# recolored.
# ---------------------------------------------------------------------------
def fig_provider_mentions(providers, top_n=15):
    df = providers.sort_values("n_threads", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(df["provider"], df["n_threads"], color=BLUE, height=0.65)
    xmax = max(df["n_threads"]) * 1.1
    for bar, val in zip(bars, df["n_threads"]):
        ax.text(val + xmax * 0.01, bar.get_y() + bar.get_height() / 2, f"{val:,}",
                 va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
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
    # 'unclear'/'too_short' bars run tall on the right; build-/buy-leaning
    # stay low on the left, so the legend sits there instead of over data.
    ax.legend(frameon=False, loc="upper left", fontsize=9)
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
              "n = threads in that stance group with >=1 qualifying decision sentence "
              "(tier-1 only) — read as exploratory, not conclusive.",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig6_sentiment_by_stance")


def main():
    master = pd.read_csv(os.path.join(OUT_DIR, "threads_master.csv"))
    factors = pd.read_csv(os.path.join(OUT_DIR, "factor_salience.csv"))
    providers = pd.read_csv(os.path.join(OUT_DIR, "provider_mentions.csv"))

    print("Generating figures from out/ (read-only) ...")
    fig_corpus_by_subreddit(master)
    fig_build_vs_buy_presence(master)
    fig_factor_salience(factors)
    fig_provider_mentions(providers)
    fig_stance_by_tier(master)
    fig_sentiment_by_stance(master)
    print(f"Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
