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
thesis: slot 1 maroon = build, slot 2 terracotta = buy, slot 3 amber = mixed,
gray = unclear/too_short. Single-series ranked bars use one flat hue
(sequential maroon) since the bar length already encodes magnitude.
Palette is built from Bucerius Law School's brand red; slots validated
with the dataviz skill's validate_palette.py (lightness band, chroma
floor, CVD separation, normal-vision floor all pass; contrast on amber
is the same pre-existing WARN as before, mitigated by direct value labels).
"""

import json
import os
import textwrap

import circlify
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

MAROON = "#8f202f"      # slot 1 — build-leaning (Bucerius maroon)
TERRACOTTA = "#c46a3f"     # slot 2 — buy-leaning (terracotta)
YELLOW = "#eda100"    # slot 3 — mixed (amber)
GRAY_UNCLEAR = "#c3c2b7"
GRAY_SHORT = "#e1e0d9"

STANCE_COLORS = {
    "build-leaning": MAROON,
    "buy-leaning": TERRACOTTA,
    "mixed": YELLOW,
    "unclear": GRAY_UNCLEAR,
    "too_short": GRAY_SHORT,
}
STANCE_ORDER = ["build-leaning", "buy-leaning", "mixed", "unclear", "too_short"]

# ---- reference counts: "how many submissions were EVER posted in this
# subreddit / this year" — the denominator for "what share of a subreddit's
# threads matched our query", not filtered by query or tier. ----------------
def _subreddit_total_posts():
    """Per-subreddit total, read straight from step 1's own collection stats
    (out/steps/step1_run_params.json) so it always matches whatever is
    currently in data/ — no dict to remember to update by hand when a
    subreddit is added (a hardcoded copy here previously went stale and
    silently dropped subreddits from fig 1 instead of erroring)."""
    params_path = os.path.join(OUT_DIR, "steps", "step1_run_params.json")
    with open(params_path, encoding="utf-8") as fh:
        params = json.load(fh)
    return {sub: s["posts_total"] for sub, s in params["stats"].items()}


# Total submissions posted per calendar year, summed across all corpus
# subreddits. Unlike the per-subreddit total above, step 1 doesn't break its
# stats down by year, so this one is still a captured snapshot — re-scan
# created_utc across every data/*_posts.jsonl (dedup by post id) and update
# below if data/ is refreshed with newer dumps or new subreddits. 2026 is
# partial: the dumps cover through 2026-06-29.
SUBREDDIT_TOTAL_POSTS_BY_YEAR = {
    2022: 33188,
    2023: 47643,
    2024: 68256,
    2025: 78173,
    2026: 34173,   # partial year, through 2026-06-29
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
# Figure 1 — query yield by subreddit (percentage view)
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# Metric: threads kept (post query + tier filter) as a % of ALL threads ever
# posted in that subreddit — not a raw count, so a small subreddit that is
# heavily build-vs-buy focused isn't visually dwarfed by a large one that
# mostly isn't. Axis fixed at 0-100% (the true scale of a %) rather than
# 1.15x the max bar. Subreddits below 1% are dropped from the chart entirely
# (no "Other" bar). No gridlines: the direct % label on every bar already
# carries the value.
# ---------------------------------------------------------------------------
def fig_corpus_by_subreddit_pct(master):
    kept = master["subreddit"].value_counts()
    totals = pd.Series(_subreddit_total_posts())
    df = pd.DataFrame({"kept": kept, "total": totals.reindex(kept.index)})
    df["pct"] = df["kept"] / df["total"] * 100

    excluded = df[df["pct"] < 1.0].sort_values("pct", ascending=False)
    large = df[df["pct"] >= 1.0].sort_values("pct")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(large.index, large["pct"], color=MAROON, height=0.65)
    for bar, val in zip(bars, large["pct"]):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Threads kept (% of subreddit's total posts)")
    ax.set_title("Query yield by subreddit", loc="left", fontsize=11, pad=12)
    ax.set_xlim(0, 100)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    if len(excluded):
        ax.text(0.98, 0.05, "Excluded (<1% yield):\n" + ", ".join(excluded.index),
                 transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                 color=INK_SECONDARY,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=BASELINE, linewidth=0.8))
    _save(fig, "fig1_corpus_by_subreddit")


# ---------------------------------------------------------------------------
# Figure 1b — kept threads by subreddit (raw material contributed)
# Job: compare magnitude, ranked -> horizontal bar, single sequential hue.
# Companion to fig 1: fig 1 ranks by YIELD RATE (%), which rewards a subreddit
# for being a large fraction of ITSELF regardless of scale — a 5-post
# subreddit at 60% (r/LegalAIPrompts) outranks r/law's 0.1%. This figure
# ranks subreddits by RAW kept-thread count instead — how much material each
# one actually contributed. Subreddits under 5 kept threads are dropped from
# the chart (too small a bar to read) and named in a corner note instead.
# No gridlines: the end label already carries the value.
# ---------------------------------------------------------------------------
def fig_corpus_by_subreddit_counts(master):
    kept = master["subreddit"].value_counts()

    excluded = kept[kept < 5].sort_values(ascending=False)
    large = kept[kept >= 5].sort_values()   # ascending -> barh draws the largest at the top

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xmax = large.max() * 1.15
    bars = ax.barh(large.index, large.values, color=MAROON, height=0.65)
    for bar, kept_v in zip(bars, large.values):
        ax.text(kept_v + xmax * 0.012, bar.get_y() + bar.get_height() / 2,
                 f"{int(kept_v):,}",
                 va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Threads kept")
    ax.set_title("Kept threads by subreddit", loc="left", fontsize=11, pad=12)
    ax.set_xlim(0, xmax)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    if len(excluded):
        fig.text(0.01, -0.06, "Excluded (<5 kept threads): " + ", ".join(excluded.index),
                 fontsize=8, color=INK_SECONDARY, ha="left",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=BASELINE, linewidth=0.8))
    _save(fig, "fig1b_kept_threads_by_subreddit")


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
    ax.bar(xk, kept_solid, width, color=MAROON, zorder=3)
    ax.bar(xk, kept_remainder, width, bottom=kept_solid, color=MAROON, alpha=0.35,
           hatch="//", edgecolor="white", linewidth=0, zorder=3)
    ax.bar(xt, total_solid, width, color=INK_MUTED, zorder=3)
    ax.bar(xt, total_remainder, width, bottom=total_solid, color=INK_MUTED, alpha=0.35,
           hatch="//", edgecolor="white", linewidth=0, zorder=3)

    for xi, y in zip(xk, years):
        top = kept_proj[y]
        ax.text(xi, top * 1.08, f"{round(top):,}", ha="center", va="bottom",
                 fontsize=7.5, color=INK_SECONDARY)
    for xi, y in zip(xt, years):
        top = total_proj[y]
        ax.text(xi, top * 1.08, f"{round(top):,}", ha="center", va="bottom",
                 fontsize=7.5, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(50, max(total_proj.values()) * 2.2)
    ax.set_ylabel("Threads (log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_title("Build vs. buy discussion presence", loc="left", fontsize=11, pad=12)
    ax.set_axisbelow(True)
    handles = [
        Patch(color=MAROON, label="Threads kept"),
        Patch(color=INK_MUTED, label="Total posts"),
        Patch(facecolor="white", edgecolor=INK_MUTED, hatch="//",
              label=f"Projected remainder of {current_year}"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
             bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=8.5)
    _clean_axes(ax)
    fig.text(0.01, -0.03,
              f"{current_year} data covers Jan 1-{cutoff.strftime('%b %d')} "
              f"({cutoff.dayofyear} of {days_in_year} days); the hatched segment projects "
              "the rest of the year at the same daily pace.",
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
def fig_factor_salience(factors, top_n=None):
    df = factors.sort_values("thread_coverage_pct_all", ascending=True)
    if top_n is not None:
        df = df.tail(top_n)
    fig, ax = plt.subplots(figsize=(7.5, max(5.5, 0.38 * len(df) + 1.2)))
    bars = ax.barh(df["label"], df["thread_coverage_pct_all"], color=MAROON, height=0.65)
    for bar, val in zip(bars, df["thread_coverage_pct_all"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%",
                 va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Share of all threads mentioning the factor (%)")
    title = f"Top {top_n} factors by thread salience" if top_n is not None else "All factors by thread salience"
    ax.set_title(title, loc="left", fontsize=11, pad=12)
    ax.set_xlim(0, 100)
    _clean_axes(ax, hide_spines=("top", "right", "left"))
    ax.tick_params(left=False)
    _save(fig, "fig3_factor_salience")


# ---------------------------------------------------------------------------
# Figure 4 — provider mentions, market-presence overview
# Job: show who the biggest players are at a glance, not exact counts ->
# packed bubble chart, single sequential hue. Circle *area* (not radius) is
# proportional to n_threads (via circlify), which is the standard fix for
# the classic bubble-chart mistake of scaling radius linearly and thereby
# exaggerating differences. Exact values are deliberately omitted from the
# marks themselves — this figure is about relative market presence, not
# precise comparison (see fig_provider_mentions v1 / bar chart for that).
# ---------------------------------------------------------------------------
def fig_provider_mentions(providers, min_threads=50, title="Providers by market presence",
                           save_name="fig4_provider_mentions"):
    # Cutting providers below min_threads (rather than a fixed top-N) keeps
    # every provider whose count is high enough to render as a legible,
    # labeled circle, and drops exactly the ones that would otherwise pack in
    # as unreadable slivers. Threshold chosen empirically at 50: below that,
    # circles get small enough that some labels (esp. longer names like
    # "Power Platform", "Adobe/Acrobat Sign") no longer fit even at the 6pt
    # floor (see fig_provider_mentions v1 for the un-cut version).
    df = providers.sort_values("n_threads", ascending=False)
    kept = df[df["n_threads"] >= min_threads]
    dropped = df[df["n_threads"] < min_threads].sort_values(
        "n_threads", ascending=False)
    df = kept
    data = [{"id": row.provider, "datum": row.n_threads} for row in df.itertuples()]
    circles = circlify.circlify(
        data, show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1),
    )

    # Fit axis limits to the actual packed cluster (not a fixed -1..1 square)
    # so _save's bbox_inches="tight" doesn't leave dead white space around it
    # — the Axes' own background patch would otherwise force the crop to the
    # full nominal enclosure even where no circle reaches it.
    pad = 0.05
    x_min = min(c.x - c.r for c in circles) - pad
    x_max = max(c.x + c.r for c in circles) + pad
    y_min = min(c.y - c.r for c in circles) - pad
    y_max = max(c.y + c.r for c in circles) + pad

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()  # initializes a renderer so text extents can be measured
    renderer = fig.canvas.get_renderer()

    # Split on "/" for alt-name providers (e.g. "Claude/Anthropic") so they
    # wrap on a real word boundary instead of textwrap hyphenating mid-word.
    # Two-word names with no "/" (e.g. "Bloomberg Law") get the same
    # word-boundary treatment via textwrap, balanced across two lines —
    # a single wide line is what pushed names like that below the legibility
    # floor before min_threads was tuned to account for it.
    for c in circles:
        ax.add_patch(plt.Circle((c.x, c.y), c.r, facecolor=MAROON,
                                 edgecolor="white", linewidth=2, alpha=0.9))
        name = c.ex["id"]
        if "/" in name:
            lines = name.split("/")
        elif " " in name:
            lines = textwrap.wrap(name, width=max(len(name) // 2, 1),
                                   break_long_words=False)
        else:
            lines = [name]
        fontsize = 15.0
        label = ax.text(c.x, c.y, "\n".join(lines), ha="center", va="center",
                         fontsize=fontsize, color="white", linespacing=1.2)
        # Shrink to fit: measure actual rendered size against the circle's
        # on-screen diameter and scale down (never up) so text never
        # overflows its bubble, regardless of name length or font metrics.
        bbox = label.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = ax.transData.transform([(c.x - c.r, c.y - c.r),
                                                       (c.x + c.r, c.y + c.r)])
        diameter_px = abs(x1 - x0)
        scale = min(1.0, 0.82 * diameter_px / bbox.width,
                    0.82 * diameter_px / bbox.height)
        # Safety net: below this scale even the 6pt floor would overflow the
        # bubble. min_threads should already keep this from triggering; it
        # only guards against a future data refresh reintroducing a gap.
        if scale < 0.4:
            label.remove()
            continue
        fontsize = max(6.0, fontsize * scale)
        label.set_fontsize(fontsize)

    ax.set_title(title, loc="left", fontsize=11, pad=12)

    dropped_list = ", ".join(
        f"{row.provider} ({row.n_threads})" for row in dropped.itertuples()
    )
    caption = (
        "Circle area is proportional to the number of threads mentioning the "
        f"provider. Providers mentioned in fewer than {min_threads} threads "
        f"are cut from this chart so every remaining label stays readable; "
        f"omitted ({len(dropped)}): {dropped_list}."
    )
    ax.text(0.5, -0.03, "\n".join(textwrap.wrap(caption, width=110)),
            transform=ax.transAxes, fontsize=7.5, color=INK_MUTED,
            ha="center", va="top")
    _save(fig, save_name)


# ---------------------------------------------------------------------------
# Figure 4b — provider mentions, 2026 threads only
# Job: same as fig 4 (packed bubble, single sequential hue, area encodes
# n_threads) but scoped to threads created in 2026 only, so the market-
# presence snapshot isn't diluted by four years of prior corpus. Same
# min_threads cutoff (50) and same rendering code path as fig 4 -- only the
# input counts differ -- so the two figures stay visually identical in
# style and are safe to compare side by side. Per-thread provider mentions
# are re-derived from threads_master's "providers_mentioned" column (not
# from provider_mentions.csv, which is a single all-years aggregate) filtered
# to created_utc in 2026, counting each thread at most once per provider --
# matching step4_aggregate's provider_agg["threads"] logic exactly.
# ---------------------------------------------------------------------------
def _provider_thread_counts_for_year(master, year):
    dates = pd.to_datetime(master["created_utc"], errors="coerce", utc=True)
    subset = master[dates.dt.year == year]
    counts = {}
    for cell in subset["providers_mentioned"].dropna():
        for part in cell.split("; "):
            name = part.rsplit(" (", 1)[0]
            counts[name] = counts.get(name, 0) + 1
    return pd.DataFrame({"provider": list(counts.keys()),
                          "n_threads": list(counts.values())})


def fig_provider_mentions_2026(master, min_threads=50):
    providers_2026 = _provider_thread_counts_for_year(master, 2026)
    fig_provider_mentions(providers_2026, min_threads=min_threads,
                          title="Providers by market presence (2026 only)",
                          save_name="fig4b_provider_mentions_2026")


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
           color=MAROON)
    ax.bar([i + width / 2 for i in x], tier2.values, width, label="Discourse-tier",
           color=INK_MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("-", "-\n") if s == "build-leaning" or s == "buy-leaning"
                         else s for s in STANCE_ORDER], fontsize=9)
    ax.set_ylabel("Threads")
    ax.set_title("Stance classification by corpus tier", loc="left", fontsize=11, pad=12)
    ax.set_axisbelow(True)
    # 'unclear'/'too_short' bars run tall on the right; build-/buy-leaning
    # stay low on the left, so the legend sits there instead of over data.
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _clean_axes(ax)
    _save(fig, "fig5_stance_by_tier")


# ---------------------------------------------------------------------------
# Figure 6 — decision-tone sentiment by stance
# Job: compare distributions across a few categories -> boxplot, categorical
# color matching the same build/buy hues used in Figure 5. "Mixed" threads
# are excluded: their decision sentences argue both directions, so an
# averaged tone doesn't represent anything coherent the way it does for a
# single-direction stance.
# ---------------------------------------------------------------------------
def fig_sentiment_by_stance(master):
    df = master[master["n_decision_sentences"].fillna(0).astype(float) > 0].copy()
    df = df[df["stance_heuristic"].isin(["build-leaning", "buy-leaning"])]
    groups = [df.loc[df["stance_heuristic"] == s, "decision_tone_mean"].dropna()
              for s in ["build-leaning", "buy-leaning"]]
    labels = [f"Build-leaning\n(n={len(groups[0])})",
              f"Buy-leaning\n(n={len(groups[1])})"]
    colors = [MAROON, TERRACOTTA]

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
    ax.set_axisbelow(True)
    _clean_axes(ax)
    fig.text(0.01, -0.03,
              "n = threads in that stance group with at least one qualifying decision "
              "sentence (tier-1 only).",
              fontsize=7.5, color=INK_MUTED, ha="left")
    _save(fig, "fig6_sentiment_by_stance")


def main():
    master = pd.read_csv(os.path.join(OUT_DIR, "threads_master.csv"))
    factors = pd.read_csv(os.path.join(OUT_DIR, "factor_salience.csv"))
    providers = pd.read_csv(os.path.join(OUT_DIR, "provider_mentions.csv"))

    print("Generating figures from out/ (read-only) ...")
    fig_corpus_by_subreddit_pct(master)
    fig_corpus_by_subreddit_counts(master)
    fig_build_vs_buy_presence(master)
    fig_factor_salience(factors)
    fig_provider_mentions(providers)  # top_n=None -> all providers
    fig_provider_mentions_2026(master)
    fig_stance_by_tier(master)
    fig_sentiment_by_stance(master)
    print(f"Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
