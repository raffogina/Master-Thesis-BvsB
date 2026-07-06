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
2. **No lemmatisation.** *firm/firms*, *tool/tools*, *update/updated/updates* were counted
   as different words. v2 folds them (NLTK WordNet if installed, suffix rules otherwise).
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
GATE     keep a thread iff (strong decision phrase OR weak decision verb + software noun)
         AND (legal term OR specialist subreddit OR legal-specific provider)
MEASURE  stance heuristic (build/buy-leaning), factor mentions, provider mentions,
         VADER sentiment: original post (author satisfaction proxy), comments
         (community reaction), sentence-level tone per factor/provider
OUTPUT   threads_master.csv, factor_salience.csv, provider_mentions.csv,
         keyword_frequency.csv (legacy view), run_summary.txt
```

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
| `factor_salience.csv` | RQ(ii)/(iii): which factors appear, how often, with what tone — the core results table |
| `provider_mentions.csv` | vendor-landscape barometer (§4.2), category-level build vs buy discussion |
| `keyword_frequency.csv` | exploratory annex only (inductive view, now cleaned) |
| `run_summary.txt` | numbers for §2 and §3.1 (yield per subreddit, date range, backends) |

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
- Bird, S., Klein, E. & Loper, E. (2009). *Natural Language Processing with Python.*
  O'Reilly. — NLTK (stop words, lemmatisation). WordNet: Miller (1995), CACM 38(11).
- Factor-dictionary sources are suggested per factor in `config/keywords.json`
  (Williamson 1985; Ellram 1995; Davis 1989; Shapiro & Varian 1999; Morgan & Hunt 1994;
  Barney 1991; Lacity & Hirschheim 1993; Daneshgar et al. 2013; Dahl et al. 2024).

## Ethics & platform notes

Only public content is collected; no usernames or author identifiers are stored; report
aggregates and paraphrase rather than quote verbatim where possible (Proferes et al. 2021).
Unauthenticated endpoint limitations (rate limits, result caps) are documented in the
thesis Limitations section; raw-thread caching avoids repeated scraping of the same data.
