# Build vs. Buy — Reddit Evidence Pipeline (Master Thesis)

Empirical pipeline for the thesis *"The 'Build vs. Buy' Dilemma in Legal Departments: A
Framework for Decision-Making"*. It collects Reddit discussions where legal professionals
debate building versus buying software, and turns them into four analysable tables.

- `scraper_v2.py` — current pipeline (use this).
- `scraper_plan_b_final.py` — v1, kept unchanged for the thesis Annex / audit trail.
- `config/keywords.json` — **all dictionaries live here.** This file is methodology:
  reproduce it in the Annex and cite a source for every list you keep.

## Why v1 had to be reshaped

1. **Substring gating bug.** `"air" in title` matched *airport*, `"custom"` matched
   *customers*, `"gc"` matched anything containing those letters. This is how the
   airport-land-seizure and buy-pants threads entered the dataset. v2 matches whole
   words/phrases only.
2. **No word-form folding.** *firm/firms*, *tool/tools*, *update/updated/updates* were
   counted as different words. v2 folds all variants with a Porter stemmer (suffix rules
   as fallback if NLTK is absent) and displays each stem's most frequent surface form.
3. **Stop-word gaps.** *month, got, here, ve, re, don, https, com* etc. polluted the
   output. v2 uses a standard English stop list plus a documented domain extension, and
   strips URLs/contractions before counting.
4. **Only top-level comments were read.** v1 ignored every nested reply
   (`replies` subtree), silently discarding most of each thread. v2 recurses.
5. **~25 results per subreddit.** v1 used the default search page size and one query.
   v2 requests up to 100 per query, supports pagination and multiple queries.
6. **Raw word frequency cannot answer the research questions.** Counting frequent words
   yields salient *vocabulary*, not decision *factors* (Zipf: frequency is dominated by
   generic language). v2 measures theory-driven factor dictionaries instead, and keeps the
   frequency table only as an exploratory annex output.
7. **Thesis–code mismatches to fix in the text:** §2.1.2 claims sentiment analysis "was
   used" (it did not exist in v1); §2.1.1 describes a gate requiring Scope Word AND
   Software Provider (v1 actually required intent AND (provider OR "legal" OR specialist
   subreddit)); v1 stored no dates, scores or comment counts (no descriptive statistics
   were possible).

## Pipeline

```
COLLECT  search each subreddit for the config queries (public JSON, rate-limited,
         every thread cached to disk -> re-analysis never re-scrapes)
TIER     tier 1 "decision threads":  (strong decision phrase OR decision verb +
           software noun) AND legal-domain evidence
         tier 2 "discourse threads": no decision phrasing, but legal-domain
           evidence AND tech context AND >=1 study factor or provider
         excluded: everything else (off-topic noise)
MEASURE  stance heuristic (build/buy-leaning), factor mentions, provider mentions,
         VADER sentiment: original post (author satisfaction proxy), comments
         (community reaction), sentence-level tone per factor/provider
OUTPUT   threads_master.csv, factor_salience.csv (split by tier),
         provider_mentions.csv, term_discovery.csv, keyword_frequency.csv,
         run_summary.txt
```

Report headline factor salience on tier 1 (factors *in decision contexts* — what the
research questions ask); use tier 1+2 for the discourse-wide view and as a robustness
check. Threads that discuss factors without explicit decision language are therefore
kept, just labelled, so nothing representative of the legal-tech discourse is lost.

### Run

```bash
pip install -r requirements.txt
python3 scraper_v2.py                       # full scrape (respects rate limits, hours)
python3 scraper_v2.py --reanalyze           # rebuild outputs from cache, no network
python3 scraper_v2.py --subreddits legaltech,lawfirm --limit 50   # pilot run
```

### Outputs → thesis mapping

| File | Feeds |
|---|---|
| `threads_master.csv` | corpus description (§3.3), stance × sentiment cross-tab (satisfaction with build vs buy), manual annotation columns for validation |
| `factor_salience.csv` | RQ(ii)/(iii): which factors appear, how often, with what tone — per tier; the core results table |
| `provider_mentions.csv` | vendor-landscape barometer (§4.2), category-level build vs buy discussion |
| `term_discovery.csv` | step-B factor-gap check (see below) — do this BEFORE freezing the factor list |
| `keyword_frequency.csv` | exploratory annex only (per-thread stemmed counts, v1 continuity) |
| `run_summary.txt` | numbers for §2 and §3.1 (yield per subreddit, tier counts, date range, backends) |

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
agreement between your labels and `stance_heuristic` / sentiment labels in §2. Spot-check
the ambiguous provider names listed in `config/keywords.json` (`_known_ambiguities`).

## Method bibliography (verify before citing)

- Hutto, C.J. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment
  Analysis of Social Media Text.* Proc. ICWSM. — sentiment backend and ±0.05 thresholds.
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
