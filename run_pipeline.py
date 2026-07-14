#!/usr/bin/env python3
"""Build-vs-Buy Reddit evidence pipeline — MASTER ORCHESTRATOR.

Master Thesis: "The 'Build vs. Buy' Dilemma in Legal Departments"

Runs the backbone steps and the five independent analysis modules, with an
automatic quality gate after each one:

  self-test              every module re-analyses a frozen fixture thread and
                         must match the frozen expected model exactly
  1  COLLECT (backbone)  pipeline/step1_collect.py    -> cache + manifest
                         default source: the Arctic Shift .jsonl dumps saved
                         in ./data (every file is read; names don't matter)
  2  TIER    (backbone)  pipeline/step2_tier.py       -> thread tiers table
  3  ANALYSES (independent of each other — any can be skipped/deleted/broken
     without affecting the others; all read only the tier table + cache):
     3a stance           pipeline/step3a_stance.py    -> build/buy leaning
     3b factors          pipeline/step3b_factors.py   -> factor mentions
     3c providers        pipeline/step3c_providers.py -> provider mentions
     3d sentiment        pipeline/step3d_sentiment.py -> VADER sentiment/tones
     3e terms            pipeline/step3e_terms.py     -> stemmed term counts
  4  AGGREGATE (backbone) pipeline/step4_aggregate.py -> the final tables,
                         built from whichever analyses succeeded

Failure policy (so one error never destroys the whole run):
  - a BACKBONE failure stops the pipeline (nothing downstream is possible);
  - an ANALYSIS module that fails its self-test, crashes, or fails its output
    gate is QUARANTINED: excluded from the final tables, everything else
    continues; the run then finishes with exit code 1 and a clear summary;
  - an analysis module whose .py file you deleted is simply skipped (that is
    a supported way to drop a method from the thesis);
  - --skip-sentiment / --skip-terms / --skip-stance / --skip-factors /
    --skip-providers skip a module for one run without touching any file.

All check results go to out/checks_report.txt. Typical usage:
  python3 run_pipeline.py                      # analyse the .jsonl dumps in ./data
                                               # (default source — no network, no api key)
  python3 run_pipeline.py --reanalyze          # rebuild everything from cache
  python3 run_pipeline.py --skip-sentiment     # thesis without the sentiment method
  python3 run_pipeline.py --source archive-api # previous online collector (audit trail)
  python3 -m pipeline.step3d_sentiment --out out   # re-run one module alone

Method references: VADER sentiment (Hutto & Gilbert 2014); dictionary-based
content analysis (Krippendorff 2018; Grimmer & Stewart 2013); Porter stemming
(Porter 1980); Reddit research practice and ethics (Proferes et al. 2021).
No usernames are collected or stored.
"""

import argparse
import sys

from pipeline import (PIPELINE_VERSION, checks, step1_collect,
                      step2_tier, step4_aggregate)

ANALYSIS_ORDER = ("stance", "factors", "providers", "sentiment", "terms")


def abort(checker, stage):
    checker.write_report()
    print(f"\n🛑 {stage} failed its quality gate — pipeline stopped so the "
          f"error cannot cascade into later steps.")
    print("   See out/checks_report.txt for the failing check(s); fix the "
          "cause and re-run (use --reanalyze to avoid re-scraping).")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    step1_collect.add_cli_arguments(parser)      # collection + shared flags
    parser.add_argument("--sentiment", choices=["auto", "vader", "textblob", "none"],
                        default="auto",
                        help="sentiment backend ('none' skips the sentiment module)")
    parser.add_argument("--no-legacy", action="store_true",
                        help="skip the per-thread keyword_frequency.csv output")
    for module in ANALYSIS_ORDER:
        parser.add_argument(f"--skip-{module}", action="store_true",
                            help=f"do not run the {module} analysis module this run")
    parser.add_argument("--skip-checks", action="store_true",
                        help="run without the automatic quality gates (not recommended)")
    args = parser.parse_args()

    skipped = {m for m in ANALYSIS_ORDER if getattr(args, f"skip_{m}")}
    if args.sentiment == "none":
        skipped.add("sentiment")

    if args.reanalyze:
        source_desc = "offline re-analysis of the existing cache"
    elif args.source == "archive-api":
        source_desc = "live Arctic Shift archive api"
    else:
        source_desc = f"local .jsonl dumps in '{args.data_dir}' (no network)"
    print(f"🚀 Build-vs-Buy pipeline {PIPELINE_VERSION}")
    print(f"   source : {source_desc}{' (checks OFF)' if args.skip_checks else ''}")
    if skipped:
        print(f"   skipped   : {', '.join(sorted(skipped))} (by flag)")

    checker = checks.Checker(args.out, args.config)
    quarantined = {}     # module -> reason (problems, not choices)
    absent = set()       # module file deleted (a supported choice)

    # Gate 0: prove the analysis code still behaves exactly as frozen, before
    # touching any data (and before spending any of the request budget).
    if not args.skip_checks:
        statuses = checker.self_test(args.sentiment)
        if statuses.get("tier") != "ok":
            abort(checker, "Self-test (tier rule)")
        for module in ANALYSIS_ORDER:
            if module in skipped:
                continue
            if statuses.get(module) == "absent":
                absent.add(module)
            elif statuses.get(module) == "failed":
                quarantined[module] = "failed its self-test"

    # Backbone: collect + tier (a failure here stops everything).
    step1_collect.run(args.config, args.out, args.cache,
                      reanalyze=args.reanalyze, args=args)
    if not args.skip_checks and not checker.check_step1():
        abort(checker, "Step 1 (collect)")

    step2_tier.run(args.config, args.out)
    if not args.skip_checks and not checker.check_step2():
        abort(checker, "Step 2 (tier)")

    # Analysis modules: independent — run each, quarantine on any problem.
    healthy = []
    for module in ANALYSIS_ORDER:
        if module in skipped or module in absent or module in quarantined:
            continue
        mod, status = checks.import_analysis_module(module)
        if status == "absent":
            absent.add(module)
            print(f"   ℹ️ {module} module file absent — skipped")
            continue
        if mod is None:
            quarantined[module] = status
            print(f"   ⚠️ {module} module {status} — quarantined, continuing without it")
            continue
        try:
            if module == "sentiment":
                mod.run(args.config, args.out, args.sentiment)
            else:
                mod.run(args.config, args.out)
        except Exception as exc:
            quarantined[module] = f"crashed while running: {exc!r}"
            print(f"   ⚠️ {module} module crashed ({exc!r}) — quarantined, "
                  f"continuing without it")
            continue
        if not args.skip_checks and not checker.check_module(module):
            quarantined[module] = "output failed its quality gate"
            print(f"   ⚠️ {module} output failed its quality gate — quarantined, "
                  f"its results will NOT be used")
            continue
        healthy.append(module)

    # Backbone: aggregate consumes only the healthy analyses.
    step4_aggregate.run(args.config, args.out, no_legacy=args.no_legacy,
                        use_modules=set(healthy))
    if not args.skip_checks and not checker.check_step4(set(healthy),
                                                        no_legacy=args.no_legacy):
        abort(checker, "Step 4 (aggregate)")

    if not args.skip_checks:
        checker.write_report()

    for module in sorted(skipped | absent):
        print(f"   ℹ️ {module}: not run ({'skipped by flag' if module in skipped else 'module file absent'}) "
              f"— its columns/tables are empty or unwritten")
    if quarantined:
        print("\n⚠️  Finished PARTIALLY — these analysis modules had problems and "
              "were excluded from the final tables:")
        for module, reason in quarantined.items():
            print(f"   - {module}: {reason}")
        print("   Fix the module (or delete it / use --skip-… if unwanted) and "
              "re-run with --reanalyze; every other output above is valid.")
        sys.exit(1)
    print("🎉 Done." + ("" if args.skip_checks else " All quality gates passed."))


if __name__ == "__main__":
    main()
