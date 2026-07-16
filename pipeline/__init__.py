"""Build-vs-Buy Reddit evidence pipeline, modular edition.

The former single-file scraper_v2.py is split into one module per pipeline
step. Each step reads the previous step's output file(s) from disk, does its
own work, and writes its own output file(s) — so a failure in one step stops
there instead of silently corrupting everything downstream.

Backbone (required, in order):
  step1_collect    -> cache/threads/*.json + out/steps/step1_collection_manifest.csv
  step2_tier       -> out/steps/step2_thread_tiers.csv
  step4_aggregate  -> out/threads_master.csv, factor_salience.csv,
                      provider_mentions.csv, term_discovery.csv,
                      keyword_frequency.csv, run_summary.txt

Analysis modules (mutually independent; each reads only the tier table + the
cache and writes its own file — any of them can be skipped, deleted or broken
without affecting the others):
  step3a_stance    -> out/steps/step3a_stance.json
  step3b_factors   -> out/steps/step3b_factors.json
  step3c_providers -> out/steps/step3c_providers.json
  step3d_sentiment -> out/steps/step3d_sentiment.json   (the ONLY sentiment code)
  step3e_terms     -> out/steps/step3e_term_counts.json

Quality gates:
  checks           -> out/checks_report.txt (self-test + per-step output checks
                      against the expected models in tests/expected/)

Run the whole pipeline with run_pipeline.py (repo root), or any single step
standalone, e.g.:  python3 -m pipeline.step3d_sentiment --help
"""

PIPELINE_VERSION = "2.3-modular"
