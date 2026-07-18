# Build vs. Buy — Reddit Evidence Pipeline (Master Thesis)

Empirical pipeline for the thesis *"The 'Build vs. Buy' Dilemma in Legal Departments: A
Framework for Decision-Making"*. It collects Reddit discussions where legal professionals
debate building versus buying software, and turns them into four analysable tables.

- `run_pipeline.py` — **master orchestrator (use this).** Runs the steps below in order
  with an automatic quality gate after each one.
- `data/` — **the data source: save the Arctic Shift subreddit dumps here** (`.jsonl`
  files from https://arctic-shift.photon-reddit.com/download-tool, posts + comments).
  Every `*.jsonl` file in the folder is read — file names do not matter, records are
  recognised by content and comments are attached to their post via `link_id`. No api
  key, no scraping, no network: the dumps are the complete subreddit history, which also
  removes the page/result caps of the online collector (a cleaner sampling frame).
- `pipeline/` — one module per step; each is runnable on its own and writes its own output:
  - `step1_collect.py` — *backbone* — screens every post in `data/` → cache + manifest
  - `step2_tier.py` — *backbone* — corpus tiering → thread-tiers table
  - `step3a_stance.py` — *analysis* — build/buy-leaning heuristic per thread
  - `step3b_factors.py` — *analysis* — decision-factor mentions per thread
  - `step3c_providers.py` — *analysis* — provider mentions per thread
  - `step3d_sentiment.py` — *analysis* — **all sentiment code lives here and only here**
  - `step3e_terms.py` — *analysis* — stemmed term counts (step-B discovery)
  - `step4_aggregate.py` — *backbone* — the final thesis tables from whatever analyses ran
  - `checks.py` — the automatic quality gates (see "Automatic checks" below)
  - `common.py` — shared text/matching helpers (no outputs of its own)

  The five **analysis modules are independent of each other**: each reads only the tier
  table + the cached threads. Any of them can be modified, skipped for a run
  (`--skip-stance`, `--skip-terms`, …) or even deleted outright — the others still
  run and the final tables are still produced, just without that dimension (e.g. without
  the sentiment module the master table's decision-tone columns stay empty, and
  `run_summary.txt` lists it under "Analysis modules not run").
- `tests/` — frozen fixture thread + the expected-output models the checks compare against.
- Former versions (audit trail only, no longer used) remain available in git history;
  their outputs were verified byte-identical to the current modules before each was
  retired.
- `config/keywords.json` — **all dictionaries live here.** This file is methodology:
  reproduce it in the Annex and cite a source for every list you keep.

## Methodology fixes

1. **Substring gating bug.** `"air" in title` matched *airport*, `"custom"` matched
   *customers*, `"gc"` matched anything containing those letters. This is how the
   airport-land-seizure and buy-pants threads entered the dataset. The current pipeline
   matches whole words/phrases only.
2. **No word-form folding.** *firm/firms*, *tool/tools*, *update/updated/updates* were
   counted as different words. The current pipeline folds all variants with a Porter
   stemmer (suffix rules as fallback if NLTK is absent) and displays each stem's most
   frequent surface form.
3. **Stop-word gaps.** *month, got, here, ve, re, don, https, com* etc. polluted the
   output. The current pipeline uses a standard English stop list plus a documented
   domain extension, and strips URLs/contractions before counting.
4. **Only top-level comments were read.** Every nested reply (`replies` subtree) was
   ignored, silently discarding most of each thread. The current pipeline analyses the
   full comment list: the Arctic Shift dumps deliver every comment of a thread (nested
   replies included) as one flat list.
5. **~25 results per subreddit.** Collection used the default search page size and a
   single query. The current pipeline screens every post in the downloaded subreddit
   dumps against every configured query — no result cap at all.
6. **Raw word frequency cannot answer the research questions.** Counting frequent words
   yields salient *vocabulary*, not decision *factors* (Zipf: frequency is dominated by
   generic language). The current pipeline measures theory-driven factor dictionaries
   instead, and keeps the frequency table only as an exploratory annex output.
7. **Thesis–code mismatches to fix in the text:** §2.1.2 claims sentiment analysis "was
   used" (it did not exist in the original collection); §2.1.1 describes a gate requiring
   Scope Word AND Software Provider (the original gate actually required intent AND
   (provider OR "legal" OR specialist subreddit)); no dates, scores or comment counts
   were stored originally (no descriptive statistics were possible).

## Pipeline

Each step reads only the previous step's output file(s) and writes its own, so an error
stops at its own step instead of silently corrupting everything downstream:

```
SELF-TEST  pipeline/checks.py    frozen fixture thread re-analysed by EVERY module and
                                 compared with the frozen expected model — proves the
                                 analysis code is unchanged BEFORE any data is touched
1 COLLECT  step1_collect.py      reads EVERY *.jsonl file in data/ (Arctic Shift subreddit
                                 dumps; no network anywhere in this step), screens every
                                 post against the config queries + the tier rule, attaches
                                 comments via link_id, and (re)builds the matching threads
                                 in cache/threads/
                                 -> out/steps/step1_collection_manifest.csv (+ cache/)
2 TIER     step2_tier.py         tier 1 "decision threads":  (strong decision phrase OR
                                   decision verb + software noun) AND legal-domain evidence
                                 tier 2 "discourse threads": no decision phrasing, but
                                   legal-domain evidence AND tech context AND >=1 factor
                                   or provider;  tier 0/-1: excluded (with reason)
                                 -> out/steps/step2_thread_tiers.csv
3 ANALYSES (independent modules; each reads only the tier table + cache)
  3a stance     step3a_stance.py     build/buy-leaning heuristic
                                     -> out/steps/step3a_stance.json
  3b factors    step3b_factors.py    decision-factor mentions
                                     -> out/steps/step3b_factors.json
  3c providers  step3c_providers.py  provider mentions (vendor barometer)
                                     -> out/steps/step3c_providers.json
  3d sentiment  step3d_sentiment.py  mean VADER tone of the first-person past-decision
                                     sentences of each tier-1 thread (the only sentiment
                                     measured anywhere in the pipeline)
                                     -> out/steps/step3d_sentiment.json
  3e terms      step3e_terms.py      stemmed term counts for the step-B discovery check
                                     -> out/steps/step3e_term_counts.json
4 AGGREGATE step4_aggregate.py   joins whatever analyses succeeded
                                 -> out/threads_master.csv, factor_salience.csv (split by
                                 tier), provider_mentions.csv, term_discovery.csv,
                                 run_summary.txt
```

After every step its output is validated against an expected-output model
(`tests/expected/*.expected.json`); a failed check stops the pipeline (see below).

Report headline factor salience on tier 1 (factors *in decision contexts* — what the
research questions ask); use tier 1+2 for the discourse-wide view and as a robustness
check. Threads that discuss factors without explicit decision language are therefore
kept, just labelled, so nothing representative of the legal-tech discourse is lost.

### Run

```bash
pip install -r requirements.txt
# 1) download the posts + comments dumps of every study subreddit from
#    https://arctic-shift.photon-reddit.com/download-tool and save them in data/
# 2) run:
python3 run_pipeline.py                       # analyse the dumps in data/ (no network)
python3 run_pipeline.py --reanalyze           # rebuild outputs from cache (no data/ needed)
```

Subreddit selection is done by choosing which dumps to save into `data/` — every
subreddit found there is analysed; the pipeline itself does no subreddit filtering.

**Corpus hygiene:** dump collection rebuilds each matching thread in `cache/threads/`
(overwriting any older cached copy of the same thread), but threads cached by an earlier
run that do not match the current dumps stay in the cache and would be mixed in by a
later `--reanalyze`. For a corpus that is exactly "the dumps", start the first dump run
with an empty `cache/threads/` (rename the old folder to keep it) — the cache rebuilds
from `data/` in seconds.

Any step can also be re-run on its own (useful after a config change — steps 2-4 never
touch the network):

```bash
python3 -m pipeline.step2_tier --out out        # re-tier the collected threads
python3 -m pipeline.step3b_factors --out out    # re-run one analysis module alone
python3 -m pipeline.step3d_sentiment --out out  # e.g. only sentiment
python3 -m pipeline.step4_aggregate --out out   # rebuild the final tables
python3 -m pipeline.checks --out out            # re-validate whatever outputs exist
```

## Automatic checks (quality gates)

Every run is guarded by `pipeline/checks.py`; all results go to `out/checks_report.txt`
(keep it as the audit trail for §2):

1. **Self-test before anything runs.** A frozen synthetic thread
   (`tests/fixtures/fixture_thread.json`) is re-analysed **by every analysis module** with
   a frozen config snapshot and compared **exactly** against
   `tests/expected/fixture_analysis.expected.json` (tier, stance, every factor/provider
   count, the decision-sentence tone, every stem count). If a module's code was
   accidentally changed, that is caught before any data is touched.
2. **An output gate after each step.** The step's real output is compared with its
   expected-output model in `tests/expected/*.expected.json` (exact columns, minimum rows,
   unique keys, allowed values, numeric ranges such as sentiment ∈ [-1, 1] and
   percentages ∈ [0, 100]) plus cross-step consistency rules (every module covers exactly
   the threads step 2 kept; decision + discourse counts add up; dictionary names match
   the config).
3. **Backbone fail = stop; analysis fail = quarantine.** If collection, tiering or
   aggregation fails a gate, the pipeline halts immediately (nothing downstream is
   possible). If an *analysis module* fails its self-test, crashes, or fails its output
   gate, it is **quarantined**: its results are excluded, every other module still runs,
   the final tables are still produced, and the run ends with a clear "finished
   PARTIALLY" summary (exit code 1). A module whose file you deleted on purpose is simply
   noted and skipped. `--skip-checks` disables the gates (not recommended).

After an **intentional** method change (e.g. you deliberately changed the tier rule),
re-freeze the self-test model with `python3 tests/freeze_expected_fixture.py` and document
the change in the methodology. Adding factor terms in `config/keywords.json` does **not**
require re-freezing: the self-test runs on its own config snapshot.

### Outputs → thesis mapping

| File | Feeds |
|---|---|
| `threads_master.csv` | corpus description (§3.3), stance × decision-tone cross-tab (satisfaction with build vs buy), manual annotation columns for validation |
| `factor_salience.csv` | RQ(ii)/(iii): which factors appear, how often (coverage) and with what relative weight (share of all factor mentions) — per tier; the core results table |
| `provider_mentions.csv` | vendor-landscape barometer (§4.2), category-level build vs buy discussion |
| `term_discovery.csv` | step-B factor-gap check (see below) — do this BEFORE freezing the factor list |
| `run_summary.txt` | numbers for §2 and §3.1 (posts screened, yield per subreddit, tier counts, date range, backends) |

## Step B: factor-gap check (run before finalising the factor list)

Workflow: (1) draft the factor list from literature; (2) run the pipeline; (3) open
`term_discovery.csv`, filter rows with an empty `covered_by` column, and review the
high-frequency terms — each is a candidate factor the literature list might be missing
(the run summary prints the top 15); (4) add any genuine factor to the config and re-run
with `--reanalyze` (no re-scraping needed); (5) freeze the list and report the check in §2.

Why a **Porter stemmer** here instead of the WordNet lemmatiser: the discovery table must
maximise *recall* of word variants — build/builds/building/builder fold together, as do
integration/integrations/integrate. The WordNet lemmatiser without part-of-speech tagging
only folds noun plurals ("building" stays "building"), which would fragment exactly the
verb-heavy decision vocabulary this check is looking for. Over-stemming (stems like
"integr") is cosmetic only: every stem is displayed via its most frequent surface form,
and a human reviews the list anyway. Precision is unaffected because the confirmatory
factor matching (step A) uses the explicit wildcard dictionaries, not stems. Porter is the
standard, citable choice (Porter 1980); English has no grammatical-gender inflection to
handle, and if non-English subreddits are ever added, NLTK's Snowball stemmers cover
Spanish and German.

## Validation protocol (do this before writing Results)

Dictionary methods require human validation (Grimmer & Stewart 2013: "validate,
validate, validate"). Draw a random sample of ~50 gated threads, fill the
`manual_decision` (build/buy/hybrid/none) and `manual_outcome`
(satisfied/dissatisfied/mixed/unclear) columns by reading them, then report simple
agreement between your labels and the `stance_heuristic` / `decision_tone_mean`
columns in §2.

## Method bibliography (verify before citing)

- Hutto, C.J. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment
  Analysis of Social Media Text.* Proc. ICWSM. — sentiment backend.
- Grimmer, J. & Stewart, B.M. (2013). *Text as Data: The Promise and Pitfalls of Automatic
  Content Analysis Methods for Political Texts.* Political Analysis 21(3). — dictionary
  methods + validation duty.
- Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology* (4th ed.).
  Sage. — content-analysis frame for the factor dictionaries.
- Proferes, N., Jones, N., Gilbert, S., Fiesler, C. & Zimmer, M. (2021). *Studying Reddit:
  A Systematic Overview of Disciplines, Approaches, Methods, and Ethics.* Social Media +
  Society 7(2). — subreddit sampling practice and research ethics.
- Porter, M.F. (1980). *An algorithm for suffix stripping.* Program 14(3). — stemming in
  the discovery step.
- Bird, S., Klein, E. & Loper, E. (2009). *Natural Language Processing with Python.*
  O'Reilly. — NLTK implementation of stemming and stop-word handling.
- Factor-dictionary sources are suggested per factor in `config/keywords.json`
  (Williamson 1985; Ellram 1995; Davis 1989; Shapiro & Varian 1999; Morgan & Hunt 1994;
  Barney 1991; Lacity & Hirschheim 1993; Daneshgar et al. 2013; Dahl et al. 2024).

## Ethics & platform notes

Only public content is collected; no usernames or author identifiers are stored; report
aggregates and paraphrase rather than quote verbatim where possible (Proferes et al. 2021).
Unauthenticated endpoint limitations (rate limits, result caps) are documented in the
thesis Limitations section; raw-thread caching avoids repeated scraping of the same data.
