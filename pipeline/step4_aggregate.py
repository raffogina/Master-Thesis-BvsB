"""STEP 4 — AGGREGATE: build the final thesis tables.

Inputs  out/steps/step2_thread_tiers.csv     thread meta + corpus tier (required)
        out/steps/step3a_stance.json         stance module      (optional)
        out/steps/step3b_factors.json        factors module     (optional)
        out/steps/step3c_providers.json      providers module   (optional)
        out/steps/step3d_sentiment.json      sentiment module   (optional)
        out/steps/step3e_term_counts.json    terms module       (optional)
        out/steps/step1_run_params.json      parameters + per-subreddit yield
        config/keywords.json                 dictionaries (coverage column)

Every step-3 module is OPTIONAL here: this step joins whatever analysis
outputs exist and leaves the corresponding columns/tables out when a module
was skipped, deleted or failed — so one broken analysis never blocks the
others' results.

Outputs (in --out):
  threads_master.csv       always (meta + tier; analysis columns filled per module)
  factor_salience.csv      needs the factors module
  provider_mentions.csv    needs the providers module
  decision_sentences.csv   needs the sentiment module (one row per decision
                            sentence: thread_id, verb, stance, tone, sentence)
  term_discovery.csv       needs the terms module
  run_summary.txt          always

Standalone use (auto-detects which module outputs exist):
  python3 -m pipeline.step4_aggregate --config config/keywords.json --out out
"""

import argparse
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata

from . import common
from .common import compile_terms, count_matches

ALL_MODULES = ("stance", "factors", "providers", "sentiment", "terms")

MODULE_FILES = {
    "stance": common.STANCE_NAME,
    "factors": common.FACTORS_NAME,
    "providers": common.PROVIDERS_NAME,
    "sentiment": common.SENTIMENT_NAME,
    "terms": common.TERM_COUNTS_NAME,
}


class DictionaryCoverage:
    """Answers: which study dictionaries already cover a discovered term?

    Used only for the 'covered_by' column of term_discovery.csv (the step-B
    factor-gap check): uncovered high-frequency terms are candidate factors
    the literature-based list might be missing.
    """

    def __init__(self, config):
        gate = config["gate"]
        stance = config["stance"]
        self.re_strong = compile_terms(gate["strong_phrases"])
        self.re_nouns = compile_terms(gate["legaltech_terms"])
        self.re_legal = compile_terms(gate["legal_terms"])
        self.re_build = compile_terms(stance["build_signals"])
        self.re_buy = compile_terms(stance["buy_signals"])
        self.factors = common.compile_factor_patterns(config)
        self.providers = common.compile_provider_patterns(config)

    def coverage_of(self, surface_forms):
        """Which existing dictionaries already cover any of these surface forms?"""
        probe = " ".join(surface_forms)
        covered = []
        for key, (_label, pattern) in self.factors.items():
            if count_matches(pattern, probe):
                covered.append(f"factor:{key}")
        for entry in self.providers:
            name = entry[0]
            if common.count_provider_mentions(entry, probe):
                covered.append(f"provider:{name}")
        if (count_matches(self.re_strong, probe) or count_matches(self.re_build, probe)
                or count_matches(self.re_buy, probe)):
            covered.append("gate/stance")
        if count_matches(self.re_nouns, probe) or count_matches(self.re_legal, probe):
            covered.append("context-term")
        return "; ".join(covered)


def _lib_version(dist_name):
    """Installed version of a dependency, for the run_summary audit trail."""
    try:
        return metadata.version(dist_name)
    except Exception:
        return "not installed"


def _load_module_output(sdir, module, use_modules):
    """Payload of one analysis module, or None if skipped/absent/quarantined."""
    if use_modules is not None and module not in use_modules:
        return None
    path = os.path.join(sdir, MODULE_FILES[module])
    if not os.path.exists(path):
        return None
    return common.read_json(path)


def run(config_path, out_dir, use_modules=None):
    """use_modules: set of module names whose outputs may be consumed
    (the orchestrator passes the modules that ran AND passed their quality
    gate); None = standalone mode, auto-detect by file existence."""
    config = common.load_config(config_path)
    coverage = DictionaryCoverage(config)
    min_count = config["terms"]["min_count"]

    sdir = common.steps_dir(out_dir)
    tier_rows = common.read_csv(os.path.join(sdir, common.TIERS_NAME))
    kept = [r for r in tier_rows if r["corpus_tier"] in ("1", "2")]
    run_params = common.read_json(os.path.join(sdir, common.RUN_PARAMS_NAME))

    payload = {m: _load_module_output(sdir, m, use_modules) for m in ALL_MODULES}
    used = [m for m in ALL_MODULES if payload[m] is not None]
    unused = [m for m in ALL_MODULES if payload[m] is None]

    def by_id(module):
        data = payload[module]
        return ({t["thread_id"]: t for t in data["threads"]} if data else {})

    stance_of = by_id("stance")
    factors_of = {tid: t.get("factors", {}) for tid, t in by_id("factors").items()}
    providers_of = {tid: t.get("providers", {}) for tid, t in by_id("providers").items()}
    sentiment_of = by_id("sentiment")   # tier-1 threads only (decision tone)

    # ---- per-thread master table (iterates in step-2 order) ------------------
    master_rows = []
    factor_agg = {}         # key -> {"label", tiers: {1: {...}, 2: {...}}}
    provider_agg = {}       # name -> {"category", "threads", "mentions"}
    decision_sentence_rows = []  # one row per decision sentence (box-plot-ready)

    for meta in kept:
        tid = meta["thread_id"]
        tier = int(meta["corpus_tier"])
        st = stance_of.get(tid, {})
        fac = factors_of.get(tid, {})     # {key: [label, mentions]}
        prov = providers_of.get(tid, {})  # {name: [category, mentions]}
        sen = sentiment_of.get(tid, {})

        for entry in sen.get("sentences", []):
            decision_sentence_rows.append({
                "thread_id": tid,
                "subreddit": meta["subreddit"],
                "corpus_tier": tier,
                "verb": entry["verb"],
                "stance": entry["stance"],
                "tone": entry["tone"],
                "sentence": entry["sentence"],
            })

        master_rows.append({
            "subreddit": meta["subreddit"],
            "thread_id": tid,
            "thread_url": meta["thread_url"],
            "title": meta["title"],
            "created_utc": meta["created_utc"],
            "post_score": meta["post_score"],
            "num_comments_reported": meta["num_comments_reported"],
            "n_comments_scraped": meta["n_comments_scraped"],
            "corpus_tier": tier,
            "tier_reason": meta["tier_reason"],
            "stance_heuristic": st.get("stance", ""),
            "build_signal_hits": st.get("build_hits", ""),
            "buy_signal_hits": st.get("buy_hits", ""),
            "providers_mentioned": "; ".join(
                f"{n} ({v[1]})" for n, v in sorted(prov.items())),
            "factors_mentioned": "; ".join(
                f"{v[0]} ({v[1]})" for _k, v in sorted(fac.items())),
            "decision_tone_mean": sen.get("decision_tone_mean"),
            "n_decision_sentences": sen.get("n_decision_sentences", ""),
            "manual_decision": "",
            "manual_outcome": "",
            "manual_notes": "",
        })

        for key, (label, mentions) in fac.items():
            agg = factor_agg.setdefault(key, {"label": label, "tiers": {}})
            bucket = agg["tiers"].setdefault(tier, {"threads": 0, "mentions": 0})
            bucket["threads"] += 1
            bucket["mentions"] += mentions

        for name, (category, mentions) in prov.items():
            agg = provider_agg.setdefault(name, {"category": category,
                                                 "threads": 0, "mentions": 0})
            agg["threads"] += 1
            agg["mentions"] += mentions

    n_threads = len(master_rows)
    n_tier1 = sum(1 for r in master_rows if r["corpus_tier"] == 1)
    n_tier2 = n_threads - n_tier1
    print(f"📊 Step 4 — aggregating {n_threads} threads "
          f"({n_tier1} decision-tier, {n_tier2} discourse-tier).")
    if unused:
        print(f"   ⚠️ analysis modules not consumed this run: {', '.join(unused)} "
              f"(their columns/tables are left empty or unwritten)")

    if master_rows:
        common.write_csv(os.path.join(out_dir, "threads_master.csv"),
                         list(master_rows[0].keys()), master_rows)

    # ---- factor salience per tier (needs the factors module) -----------------
    # PRIMARY metric: thread_coverage_pct_* - the share of threads that
    # mention the factor AT LEAST ONCE (a thread contributes at most 1 to the
    # underlying count no matter how many times it mentions the factor, no
    # matter how long the thread is, and regardless of how many terms the
    # factor's dictionary has - see step3b_factors.py). This is what the
    # table is sorted by. mention_share_pct is a SECONDARY, informational
    # column only (share of all factor mentions; sums to ~100 across rows) -
    # it is biased by dictionary size (a 50-term dictionary racks up more
    # mentions than a 13-term one for equally salient factors) and by thread
    # length/repetition, so don't use it to rank or compare factors.
    def tier_cells(bucket, base):
        if not bucket:
            return 0, 0
        pct = round(100 * bucket["threads"] / base, 1) if base else 0
        return bucket["threads"], pct

    factor_rows = []
    if payload["factors"] is not None:
        total_factor_mentions = sum(b["mentions"] for agg in factor_agg.values()
                                    for b in agg["tiers"].values())
        for key, agg in sorted(factor_agg.items(),
                               key=lambda kv: -sum(b["threads"] for b in kv[1]["tiers"].values())):
            t1_threads, t1_pct = tier_cells(agg["tiers"].get(1), n_tier1)
            t2_threads, t2_pct = tier_cells(agg["tiers"].get(2), n_tier2)
            all_threads = t1_threads + t2_threads
            all_mentions = sum(b["mentions"] for b in agg["tiers"].values())
            factor_rows.append({
                "factor": key,
                "label": agg["label"],
                "n_threads_decision": t1_threads,
                "thread_coverage_pct_decision": t1_pct,
                "n_threads_discourse": t2_threads,
                "thread_coverage_pct_discourse": t2_pct,
                "n_threads_all": all_threads,
                "thread_coverage_pct_all": round(100 * all_threads / n_threads, 1) if n_threads else 0,
                "total_mentions": all_mentions,
                "mention_share_pct": (round(100 * all_mentions / total_factor_mentions, 1)
                                      if total_factor_mentions else 0),
            })
        common.write_csv(os.path.join(out_dir, "factor_salience.csv"),
                         ["factor", "label", "n_threads_decision", "thread_coverage_pct_decision",
                          "n_threads_discourse", "thread_coverage_pct_discourse",
                          "n_threads_all", "thread_coverage_pct_all",
                          "total_mentions", "mention_share_pct"], factor_rows)
    else:
        print("   ⚠️ factor_salience.csv NOT rewritten (factors module not run); "
              "any existing file is from an earlier run")

    # ---- provider barometer (needs the providers module) ----------------------
    # pct_of_total_mentions: same relative-weight logic as factor_salience.csv
    # (share of all provider mentions; sums to ~100 across rows).
    provider_rows = []
    if payload["providers"] is not None:
        total_provider_mentions = sum(a["mentions"] for a in provider_agg.values())
        for name, agg in sorted(provider_agg.items(),
                                key=lambda kv: (-kv[1]["threads"], -kv[1]["mentions"])):
            provider_rows.append({
                "provider": name,
                "category": agg["category"],
                "n_threads": agg["threads"],
                "pct_of_total_mentions": (round(100 * agg["mentions"] / total_provider_mentions, 1)
                                          if total_provider_mentions else 0),
                "total_mentions": agg["mentions"],
            })
        common.write_csv(os.path.join(out_dir, "provider_mentions.csv"),
                         ["provider", "category", "n_threads",
                          "pct_of_total_mentions", "total_mentions"],
                         provider_rows)
    else:
        print("   ⚠️ provider_mentions.csv NOT rewritten (providers module not run); "
              "any existing file is from an earlier run")

    # ---- decision sentences, one row per sentence (needs the sentiment module) -
    # Un-collapsed detail behind decision_tone_mean: every first-person
    # past-decision sentence in a tier-1 thread, its own VADER tone, and its
    # build/buy stance tag. Tidy/long format on purpose - e.g.
    # sns.boxplot(x="stance", y="tone", data=pd.read_csv("decision_sentences.csv"))
    # to compare the tone distribution of build-decision vs buy-decision sentences.
    if payload["sentiment"] is not None:
        common.write_csv(os.path.join(out_dir, "decision_sentences.csv"),
                         ["thread_id", "subreddit", "corpus_tier", "verb",
                          "stance", "tone", "sentence"], decision_sentence_rows)
    else:
        print("   ⚠️ decision_sentences.csv NOT rewritten (sentiment module not run); "
              "any existing file is from an earlier run")

    # ---- step-B discovery table (needs the terms module) ----------------------
    discovery_rows = []
    if payload["terms"] is not None:
        stem_total = Counter()
        stem_threads = Counter()
        stem_surfaces = {}
        for record in payload["terms"]["threads"]:
            for stem, count in record["counts"].items():
                stem_total[stem] += count
                stem_threads[stem] += 1
                stem_surfaces.setdefault(stem, Counter()).update(record["surfaces"][stem])

        display = {stem: surfaces.most_common(1)[0][0]
                   for stem, surfaces in stem_surfaces.items()}
        for stem, total in stem_total.most_common():
            if total < min_count:
                continue
            top_surfaces = [s for s, _ in stem_surfaces[stem].most_common(3)]
            discovery_rows.append({
                "stem": stem,
                "display_form": display[stem],
                "surface_forms": "; ".join(top_surfaces),
                "total_count": total,
                "n_threads": stem_threads[stem],
                "pct_of_threads": round(100 * stem_threads[stem] / n_threads, 1) if n_threads else 0,
                "covered_by": coverage.coverage_of(top_surfaces),
            })
        common.write_csv(os.path.join(out_dir, "term_discovery.csv"),
                         ["stem", "display_form", "surface_forms", "total_count",
                          "n_threads", "pct_of_threads", "covered_by"], discovery_rows)
    else:
        print("   ⚠️ term_discovery.csv NOT rewritten "
              "(terms module not run); any existing file is from an earlier run")

    # ---- run summary (for the Methodology / Results sections) ------------------
    # step1's stats[sub]["tier1"/"tier2"] are a title+body-only PRE-FILTER
    # count (step1_collect.py's TierAssigner call): they do not see how many
    # comments were actually scraped, so they do not reflect step 2's
    # comment-less-thread demotion (a tier-1 thread with 0 scraped comments
    # is demoted to tier 2 - see step2_tier.py). Recompute the per-subreddit
    # tier1/tier2 breakdown from the FINAL, post-demotion tier table so it
    # reconciles with "Threads kept" below (same source, same numbers) - only
    # posts_total/results/fetched (collection-time, unaffected by demotion)
    # still come from the step1 stats.
    final_tier_counts = {}
    for r in kept:
        bucket = final_tier_counts.setdefault(r["subreddit"], {"tier1": 0, "tier2": 0})
        bucket[f"tier{r['corpus_tier']}"] += 1

    stats = run_params.get("stats", {})
    posts_total = run_params.get("posts_total")
    sentiment_backend = (payload["sentiment"]["sentiment_backend"]
                         if payload["sentiment"] is not None
                         else "none (sentiment module not run)")
    stemmer = (payload["terms"]["stemmer"] if payload["terms"] is not None
               else "none (terms module not run)")
    summary_path = os.path.join(out_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Run finished: {datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"Collector: {run_params.get('collector', common.COLLECTOR_NAME)}\n")
        fh.write(f"Sentiment backend: {sentiment_backend}\n")
        fh.write(f"Stemmer: {stemmer}\n")
        fh.write(f"Versions: python {platform.python_version()} | "
                 f"vaderSentiment {_lib_version('vaderSentiment')} | "
                 f"nltk {_lib_version('nltk')}\n")
        if unused:
            fh.write(f"Analysis modules not run: {', '.join(unused)}\n")
        fh.write(f"Queries: {run_params.get('queries', [])}\n")
        fh.write(f"Subreddits: {run_params.get('subreddits', [])}\n")
        if posts_total:
            fh.write(f"Total posts screened: {posts_total}\n")
        fh.write(f"Threads kept: {n_threads} "
                 f"(decision-tier: {n_tier1}, discourse-tier: {n_tier2})\n")
        dates = sorted(r["created_utc"] for r in master_rows if r["created_utc"])
        if dates:
            fh.write(f"Thread date range: {dates[0]} to {dates[-1]}\n")
        if stats:
            fh.write("\nPer-subreddit yield (posts in dump | query matches | "
                     "tier1 | tier2 | threads built; tier1/tier2 are FINAL "
                     "post-demotion counts, matching 'Threads kept' above):\n")
            for sub, s in stats.items():
                sub_total = s.get("posts_total")
                rate = (f" ({round(100 * s['results'] / sub_total, 1)}% of posts)"
                        if sub_total else "")
                final = final_tier_counts.get(sub, {"tier1": 0, "tier2": 0})
                fh.write(f"  r/{sub}: {sub_total if sub_total is not None else '?'} posts | "
                         f"{s['results']} query matches{rate} | "
                         f"tier1 {final['tier1']} | tier2 {final['tier2']} | "
                         f"built {s['fetched']}\n")
        if payload["stance"] is not None:
            stance_counts = Counter(r["stance_heuristic"] for r in master_rows
                                     if r["stance_heuristic"])
            fh.write("\nStance classification (build vs buy heuristic):\n")
            for label in ("build-leaning", "buy-leaning", "mixed", "unclear", "too_short"):
                if label in stance_counts:
                    fh.write(f"  {label}: {stance_counts[label]}\n")
            too_short = stance_counts.get("too_short", 0)
            if too_short:
                min_comments = config["stance"].get("min_comments", 0)
                fh.write(f"  ({too_short} threads excluded from stance classification: "
                         f"fewer than {min_comments} scraped comments, not enough text "
                         f"for the heuristic to reach its minimum hit count — see "
                         f"config stance._rule)\n")
        fh.write("\nTop factors by decision-tier coverage:\n")
        if payload["factors"] is None:
            fh.write("  (factors module not run)\n")
        for row in factor_rows[:12]:
            fh.write(f"  {row['label']}: {row['n_threads_decision']} decision threads "
                     f"({row['thread_coverage_pct_decision']}% thread coverage) | "
                     f"all: {row['n_threads_all']} ({row['thread_coverage_pct_all']}% "
                     f"thread coverage) | {row['mention_share_pct']}% mention share "
                     f"(biased by dictionary size/thread length - informational only)\n")
        fh.write("\nTop providers by thread coverage:\n")
        if payload["providers"] is None:
            fh.write("  (providers module not run)\n")
        for row in provider_rows[:15]:
            fh.write(f"  {row['provider']} [{row['category']}]: {row['n_threads']} threads, "
                     f"{row['total_mentions']} mentions "
                     f"({row['pct_of_total_mentions']}% of provider mentions)\n")
        fh.write("\nTop UNCOVERED candidate terms (factor-gap check, step B):\n")
        if payload["terms"] is None:
            fh.write("  (terms module not run)\n")
        shown = 0
        for row in discovery_rows:
            if row["covered_by"]:
                continue
            fh.write(f"  {row['display_form']} ({row['surface_forms']}): "
                     f"{row['total_count']} mentions in {row['n_threads']} threads\n")
            shown += 1
            if shown >= 15:
                break
    print(f"   💾 {summary_path}")
    return used


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/keywords.json")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
