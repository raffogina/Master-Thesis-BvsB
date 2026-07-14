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

Outputs (in --out, unchanged names/format from scraper_v2 for thesis continuity):
  threads_master.csv     always (meta + tier; analysis columns filled per module)
  factor_salience.csv    needs the factors module (tone columns need sentiment)
  provider_mentions.csv  needs the providers module (tone column needs sentiment)
  term_discovery.csv     needs the terms module
  keyword_frequency.csv  needs the terms module (skipped with --no-legacy)
  run_summary.txt        always

Standalone use (auto-detects which module outputs exist):
  python3 -m pipeline.step4_aggregate --config config/keywords.json --out out
"""

import argparse
import os
from collections import Counter
from datetime import datetime, timezone

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
        self.re_nouns = compile_terms(gate["software_nouns"])
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
        for name, _cat, _ls, pattern in self.providers:
            if count_matches(pattern, probe):
                covered.append(f"provider:{name}")
        if (count_matches(self.re_strong, probe) or count_matches(self.re_build, probe)
                or count_matches(self.re_buy, probe)):
            covered.append("gate/stance")
        if count_matches(self.re_nouns, probe) or count_matches(self.re_legal, probe):
            covered.append("context-term")
        return "; ".join(covered)


def _load_module_output(sdir, module, use_modules):
    """Payload of one analysis module, or None if skipped/absent/quarantined."""
    if use_modules is not None and module not in use_modules:
        return None
    path = os.path.join(sdir, MODULE_FILES[module])
    if not os.path.exists(path):
        return None
    return common.read_json(path)


def run(config_path, out_dir, no_legacy=False, use_modules=None):
    """use_modules: set of module names whose outputs may be consumed
    (the orchestrator passes the modules that ran AND passed their quality
    gate); None = standalone mode, auto-detect by file existence."""
    config = common.load_config(config_path)
    coverage = DictionaryCoverage(config)
    min_count = config["legacy_keywords"]["min_count"]

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
    sentiment_of = by_id("sentiment")

    # ---- per-thread master table (iterates in step-2 order) ------------------
    master_rows = []
    factor_agg = {}         # key -> {"label", tiers: {1: {...}, 2: {...}}}
    provider_agg = {}       # name -> {"category", "threads", "t1_threads", "mentions", "tones"}

    for meta in kept:
        tid = meta["thread_id"]
        tier = int(meta["corpus_tier"])
        st = stance_of.get(tid, {})
        fac = factors_of.get(tid, {})     # {key: [label, mentions]}
        prov = providers_of.get(tid, {})  # {name: [category, mentions]}
        sen = sentiment_of.get(tid, {})
        factor_tones = sen.get("factor_tones", {})
        provider_tones = sen.get("provider_tones", {})

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
            "op_sentiment": sen.get("op_sentiment"),
            "op_sentiment_label": sen.get("op_label", ""),
            "comments_sentiment_mean": sen.get("comments_sentiment"),
            "comments_sentiment_label": sen.get("comments_label", ""),
            "manual_decision": "",
            "manual_outcome": "",
            "manual_notes": "",
        })

        for key, (label, mentions) in fac.items():
            agg = factor_agg.setdefault(key, {"label": label, "tiers": {}})
            bucket = agg["tiers"].setdefault(tier, {"threads": 0, "mentions": 0, "tones": []})
            bucket["threads"] += 1
            bucket["mentions"] += mentions
            tone = factor_tones.get(key)
            if tone is not None:
                bucket["tones"].append(tone)

        for name, (category, mentions) in prov.items():
            agg = provider_agg.setdefault(name, {"category": category, "threads": 0,
                                                 "t1_threads": 0, "mentions": 0, "tones": []})
            agg["threads"] += 1
            agg["t1_threads"] += 1 if tier == 1 else 0
            agg["mentions"] += mentions
            tone = provider_tones.get(name)
            if tone is not None:
                agg["tones"].append(tone)

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
    def tier_cells(bucket, base):
        if not bucket:
            return 0, 0, ""
        tones = bucket["tones"]
        tone = round(sum(tones) / len(tones), 4) if tones else ""
        pct = round(100 * bucket["threads"] / base, 1) if base else 0
        return bucket["threads"], pct, tone

    factor_rows = []
    if payload["factors"] is not None:
        for key, agg in sorted(factor_agg.items(),
                               key=lambda kv: -sum(b["threads"] for b in kv[1]["tiers"].values())):
            t1 = agg["tiers"].get(1)
            t2 = agg["tiers"].get(2)
            t1_threads, t1_pct, t1_tone = tier_cells(t1, n_tier1)
            t2_threads, t2_pct, t2_tone = tier_cells(t2, n_tier2)
            all_threads = t1_threads + t2_threads
            all_mentions = sum(b["mentions"] for b in agg["tiers"].values())
            all_tones = [t for b in agg["tiers"].values() for t in b["tones"]]
            factor_rows.append({
                "factor": key,
                "label": agg["label"],
                "n_threads_decision": t1_threads,
                "pct_decision_threads": t1_pct,
                "tone_decision": t1_tone,
                "n_threads_discourse": t2_threads,
                "pct_discourse_threads": t2_pct,
                "tone_discourse": t2_tone,
                "n_threads_all": all_threads,
                "pct_all_threads": round(100 * all_threads / n_threads, 1) if n_threads else 0,
                "total_mentions": all_mentions,
                "tone_all": round(sum(all_tones) / len(all_tones), 4) if all_tones else "",
            })
        common.write_csv(os.path.join(out_dir, "factor_salience.csv"),
                         ["factor", "label", "n_threads_decision", "pct_decision_threads",
                          "tone_decision", "n_threads_discourse", "pct_discourse_threads",
                          "tone_discourse", "n_threads_all", "pct_all_threads",
                          "total_mentions", "tone_all"], factor_rows)
    else:
        print("   ⚠️ factor_salience.csv NOT rewritten (factors module not run); "
              "any existing file is from an earlier run")

    # ---- provider barometer (needs the providers module) ----------------------
    provider_rows = []
    if payload["providers"] is not None:
        for name, agg in sorted(provider_agg.items(),
                                key=lambda kv: (-kv[1]["threads"], -kv[1]["mentions"])):
            tones = agg["tones"]
            provider_rows.append({
                "provider": name,
                "category": agg["category"],
                "n_threads": agg["threads"],
                "n_threads_decision": agg["t1_threads"],
                "pct_of_threads": round(100 * agg["threads"] / n_threads, 1) if n_threads else 0,
                "total_mentions": agg["mentions"],
                "mean_sentence_sentiment": round(sum(tones) / len(tones), 4) if tones else "",
            })
        common.write_csv(os.path.join(out_dir, "provider_mentions.csv"),
                         ["provider", "category", "n_threads", "n_threads_decision",
                          "pct_of_threads", "total_mentions", "mean_sentence_sentiment"],
                         provider_rows)
    else:
        print("   ⚠️ provider_mentions.csv NOT rewritten (providers module not run); "
              "any existing file is from an earlier run")

    # ---- step-B discovery tables (need the terms module) ----------------------
    discovery_rows = []
    if payload["terms"] is not None:
        stem_total = Counter()
        stem_threads = Counter()
        stem_surfaces = {}
        keyword_rows_raw = []   # (sub, url, stem, count) -> display form filled later
        for record in payload["terms"]["threads"]:
            for stem, count in record["counts"].items():
                stem_total[stem] += count
                stem_threads[stem] += 1
                stem_surfaces.setdefault(stem, Counter()).update(record["surfaces"][stem])
            if not no_legacy:
                for stem, count in record["counts"].items():
                    if count >= min_count:
                        keyword_rows_raw.append((record["subreddit"], record["thread_url"],
                                                 stem, count))

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

        if not no_legacy:
            keyword_rows = [{"subreddit": s, "thread_url": u,
                             "term": display.get(stem, stem), "stem": stem, "count": c}
                            for s, u, stem, c in sorted(keyword_rows_raw,
                                                        key=lambda r: (r[0], r[1], -r[3]))]
            common.write_csv(os.path.join(out_dir, "keyword_frequency.csv"),
                             ["subreddit", "thread_url", "term", "stem", "count"], keyword_rows)
    else:
        print("   ⚠️ term_discovery.csv / keyword_frequency.csv NOT rewritten "
              "(terms module not run); any existing files are from an earlier run")

    # ---- run summary (for the Methodology / Results sections) ------------------
    stats = run_params.get("stats", {})
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
        if unused:
            fh.write(f"Analysis modules not run: {', '.join(unused)}\n")
        fh.write(f"Queries: {run_params.get('queries', [])}\n")
        fh.write(f"Subreddits: {run_params.get('subreddits', [])}\n")
        fh.write(f"Threads kept: {n_threads} "
                 f"(decision-tier: {n_tier1}, discourse-tier: {n_tier2})\n")
        dates = sorted(r["created_utc"] for r in master_rows if r["created_utc"])
        if dates:
            fh.write(f"Thread date range: {dates[0]} to {dates[-1]}\n")
        if stats:
            fh.write("\nPer-subreddit yield (unique results / tier1 / tier2 / downloaded):\n")
            for sub, s in stats.items():
                fh.write(f"  r/{sub}: {s['results']} / {s['tier1']} / "
                         f"{s['tier2']} / {s['fetched']}\n")
        fh.write("\nTop factors by decision-tier coverage:\n")
        if payload["factors"] is None:
            fh.write("  (factors module not run)\n")
        for row in factor_rows[:12]:
            fh.write(f"  {row['label']}: {row['n_threads_decision']} decision threads "
                     f"({row['pct_decision_threads']}%), tone {row['tone_decision']} "
                     f"| all: {row['n_threads_all']} ({row['pct_all_threads']}%)\n")
        fh.write("\nTop providers by thread coverage:\n")
        if payload["providers"] is None:
            fh.write("  (providers module not run)\n")
        for row in provider_rows[:15]:
            fh.write(f"  {row['provider']} [{row['category']}]: {row['n_threads']} threads, "
                     f"{row['total_mentions']} mentions, tone {row['mean_sentence_sentiment']}\n")
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
    parser.add_argument("--no-legacy", action="store_true",
                        help="skip the per-thread keyword_frequency.csv output")
    args = parser.parse_args()
    run(args.config, args.out, no_legacy=args.no_legacy)


if __name__ == "__main__":
    main()
