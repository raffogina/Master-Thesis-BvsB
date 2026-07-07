#!/usr/bin/env python3
"""
Build-vs-Buy Reddit evidence pipeline (v2)
Master Thesis: "The 'Build vs. Buy' Dilemma in Legal Departments"

Pipeline:
  1. COLLECT   search each subreddit with the queries in config/keywords.json via
               Reddit's public JSON endpoints (rate-limited, retried, cached to disk
               so analysis can be re-run offline without re-scraping).
  2. TIER      assign each thread to a corpus tier (documented in config "gate"):
                 tier 1  explicit decision context AND legal-domain evidence
                 tier 2  no explicit decision context, but legal-domain evidence,
                         a tech context, and at least one study factor or provider
                 excluded otherwise (off-topic noise)
  3. MEASURE   per kept thread: decision-stance heuristic (build/buy leaning),
               decision-factor mentions, provider mentions, and VADER sentiment
               (original post = author-satisfaction proxy; comments = community
               reaction; sentence-level tone per factor and provider).
  4. AGGREGATE factor salience per tier, provider barometer, per-thread master
               table with empty manual-annotation columns, and Porter-stemmed
               term-discovery tables for the factor-gap check (step B).

Outputs (in --out, default ./out):
  threads_master.csv     one row per kept thread (incl. corpus_tier)
  factor_salience.csv    factor x coverage x tone, split by tier
  provider_mentions.csv  provider x coverage x tone
  term_discovery.csv     stemmed corpus vocabulary with dictionary-coverage flags
                         (review 'uncovered' high-frequency terms BEFORE freezing
                         the factor list - the deductive/inductive cross-check)
  keyword_frequency.csv  per-thread stemmed counts (annex continuity with v1)
  run_summary.txt        parameters + per-subreddit yield (for the Methodology section)

Method references: VADER sentiment (Hutto & Gilbert 2014); dictionary-based content
analysis (Krippendorff 2018; Grimmer & Stewart 2013); Porter stemming (Porter 1980);
Reddit research practice and ethics (Proferes et al. 2021).
No usernames are collected or stored.
"""

import argparse
import csv
import html
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import requests

DEFAULT_USER_AGENT = "python:build-vs-buy-thesis-scraper:v2.1 (academic research; MLB thesis)"

# Minimal standard English stop list (union of the common core of the NLTK and
# scikit-learn lists); extended at runtime by config legacy_keywords.extra_stopwords.
BASE_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we what when
where which while who whom why will with you your yours yourself yourselves
would shall may might must ought
""".split())


# ---------------------------------------------------------------------------
# Term matching: phrases with optional trailing-* wildcards, space==hyphen
# ---------------------------------------------------------------------------

def compile_terms(terms):
    """Compile a list of dictionary terms into one alternation regex.

    'customiz*'  -> matches customize, customization, ...
    'in house'   -> matches 'in house', 'in-house', 'in  house'
    All matches are word-bounded to avoid the v1 substring bug
    ('air' matching 'airport', 'custom' matching 'customers').
    """
    parts = []
    for term in terms:
        term = term.strip().lower()
        if not term:
            continue
        wildcard = term.endswith("*")
        if wildcard:
            term = term[:-1]
        piece = re.escape(term).replace(r"\ ", r"[\s\-]+")
        piece = piece + (r"\w*" if wildcard else "")
        parts.append(piece)
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def count_matches(pattern, text):
    return len(pattern.findall(text)) if pattern else 0


# ---------------------------------------------------------------------------
# Stemming (step-B discovery: folds plural/singular and verb/derived forms so
# every variant of a word is summed together before human review)
# ---------------------------------------------------------------------------

def build_stemmer():
    """Prefer NLTK's Porter stemmer (Porter 1980, no corpus downloads needed);
    fall back to conservative suffix rules if NLTK is not installed.

    A stemmer is used here instead of the WordNet lemmatiser on purpose: the
    discovery table must maximise recall of word variants (build/builds/
    building/builder all fold together), while the WordNet lemmatiser without
    POS tagging only folds noun plurals ('building' stays 'building').
    Over-stemmed forms are never shown raw: each stem is displayed through its
    most frequent surface form in the corpus.
    """
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return stemmer.stem, "NLTK PorterStemmer (Porter 1980)"
    except Exception:
        def fold(word):
            for suffix in ("ations", "ation", "ings", "ing", "ers", "er",
                           "edly", "ed", "ies", "es", "s"):
                if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                    stem = word[: -len(suffix)]
                    return stem + "y" if suffix == "ies" else stem
            return word
        return fold, "suffix-rules (install nltk for Porter stemming)"


# ---------------------------------------------------------------------------
# Sentiment backends: vaderSentiment -> NLTK VADER -> TextBlob -> none
# ---------------------------------------------------------------------------

def build_sentiment(choice):
    """Return (score_fn, backend_name); score_fn maps text -> [-1, 1]."""
    def try_vader_pkg():
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return lambda t: analyzer.polarity_scores(t)["compound"], "vaderSentiment (Hutto & Gilbert 2014)"

    def try_nltk_vader():
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            analyzer = SentimentIntensityAnalyzer()
        return lambda t: analyzer.polarity_scores(t)["compound"], "NLTK VADER (Hutto & Gilbert 2014)"

    def try_textblob():
        from textblob import TextBlob
        return lambda t: TextBlob(t).sentiment.polarity, "TextBlob PatternAnalyzer (Loria 2018)"

    attempts = {
        "vader": [try_vader_pkg, try_nltk_vader],
        "textblob": [try_textblob],
        "none": [],
        "auto": [try_vader_pkg, try_nltk_vader, try_textblob],
    }[choice]
    for attempt in attempts:
        try:
            return attempt()
        except Exception:
            continue
    if choice != "none":
        print("⚠️  No sentiment library available (pip install vaderSentiment). "
              "Sentiment columns will be empty.")
    return None, "none"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"[a-z']+")


def clean_text(raw):
    text = html.unescape(raw or "")
    text = MD_LINK_RE.sub(r"\1", text)   # keep link label, drop URL
    text = URL_RE.sub(" ", text)
    text = text.replace("’", "'").replace("&#x200B;", " ")
    return text


def tokenize(text):
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        tok = tok.strip("'").replace("'", "")
        if tok:
            tokens.append(tok)
    return tokens


def sentences_of(text):
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Reddit access (public JSON endpoints) with caching
# ---------------------------------------------------------------------------

class RedditClient:
    def __init__(self, cache_dir, sleep_thread, sleep_sub, user_agent):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.cache_dir = cache_dir
        self.sleep_thread = sleep_thread
        self.sleep_sub = sleep_sub
        os.makedirs(os.path.join(cache_dir, "threads"), exist_ok=True)

    def _get_json(self, url, params=None):
        """GET with retry on 429 (rate limit) and connection drops."""
        for attempt in range(4):
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.RequestException:
                wait = 10 * (attempt + 1)
                print(f"   🚨 Network drop. Retrying in {wait}s ...")
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                print("   ⚠️ Rate-limited (429). Cooling down 70s ...")
                time.sleep(70)
                continue
            if resp.status_code != 200:
                return None, resp.status_code
            try:
                return resp.json(), 200
            except ValueError:
                return None, -1
        return None, -2

    def search(self, subreddit, query, limit, pages):
        """Yield search-result posts (dicts) for one subreddit+query."""
        after = None
        for _ in range(pages):
            params = {"q": query, "restrict_sr": "on", "sort": "relevance",
                      "t": "all", "limit": min(limit, 100)}
            if after:
                params["after"] = after
            data, status = self._get_json(
                f"https://www.reddit.com/r/{subreddit}/search.json", params)
            if data is None:
                print(f"   ⚠️ Search skipped for r/{subreddit} (status {status})")
                return
            children = data.get("data", {}).get("children", [])
            for child in children:
                if child.get("kind") == "t3":
                    yield child["data"]
            after = data.get("data", {}).get("after")
            if not after:
                return
            time.sleep(self.sleep_thread)

    def fetch_thread(self, subreddit, thread_id, permalink):
        """Fetch (or load from cache) the full thread JSON listing."""
        cache_path = os.path.join(self.cache_dir, "threads", f"{thread_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                return json.load(fh)
        time.sleep(self.sleep_thread)
        url = f"https://www.reddit.com{permalink}.json"
        data, status = self._get_json(url)
        if not isinstance(data, list) or len(data) < 2:
            return None
        wrapper = {
            "subreddit": subreddit,
            "thread_id": thread_id,
            "thread_url": f"https://www.reddit.com{permalink}",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "listing": data,
        }
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(wrapper, fh)
        return wrapper


def walk_comments(children):
    """Recursively yield comment bodies (v1 only read top-level comments)."""
    for child in children or []:
        if child.get("kind") != "t1":
            continue
        data = child.get("data", {})
        body = data.get("body")
        if body and body not in ("[deleted]", "[removed]"):
            yield body
        replies = data.get("replies")
        if isinstance(replies, dict):
            yield from walk_comments(replies.get("data", {}).get("children"))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class Analyzer:
    def __init__(self, config, sentiment_choice):
        gate = config["gate"]
        self.re_strong = compile_terms(gate["strong_phrases"])
        self.re_weak = compile_terms(gate["weak_verbs"])
        self.re_nouns = compile_terms(gate["software_nouns"])
        self.re_legal = compile_terms(gate["legal_terms"])
        self.specialist_subs = {s.lower() for s in config["subreddits"]["specialist"]}

        stance = config["stance"]
        self.re_build = compile_terms(stance["build_signals"])
        self.re_buy = compile_terms(stance["buy_signals"])
        self.stance_min = stance.get("min_hits", 3)
        self.stance_ratio = stance.get("dominance_ratio", 2.0)

        self.factors = {}
        for key, spec in config["factors"].items():
            if key.startswith("_"):
                continue
            self.factors[key] = (spec["label"], compile_terms(spec["terms"]))

        self.providers = []
        for entry in config["providers"]["entries"]:
            self.providers.append((entry["name"], entry["category"],
                                   bool(entry.get("legal_specific")),
                                   compile_terms(entry["patterns"])))

        self.score, self.sentiment_backend = build_sentiment(sentiment_choice)
        senti = config["sentiment"]
        self.pos_thr = senti["positive_threshold"]
        self.neg_thr = senti["negative_threshold"]

        legacy = config["legacy_keywords"]
        self.stopwords = BASE_STOPWORDS | set(legacy["extra_stopwords"])
        self.min_count = legacy["min_count"]
        self.min_len = legacy["min_token_len"]
        self.stem, self.stem_backend = build_stemmer()

    # -- corpus tiering ------------------------------------------------------
    def tier(self, subreddit, title, selftext):
        """Assign a corpus tier from title+body (rule documented in config).

        tier 1: explicit decision context AND legal-domain evidence
        tier 2: legal-domain evidence AND tech context AND >=1 factor/provider
        tier 0: excluded (off-topic noise)
        Tiering uses only title+body so the pre-download filter and the final
        analysis apply the identical rule.
        """
        head = clean_text(f"{title}\n{selftext}")

        provider_hit = None
        legal_provider_hit = None
        for name, _cat, legal_specific, pattern in self.providers:
            if count_matches(pattern, head):
                provider_hit = provider_hit or name
                if legal_specific:
                    legal_provider_hit = legal_provider_hit or name

        legal_domain = (bool(count_matches(self.re_legal, head))
                        or subreddit.lower() in self.specialist_subs
                        or bool(legal_provider_hit))
        if not legal_domain:
            return 0, "no legal-domain evidence"

        decision = None
        if count_matches(self.re_strong, head):
            decision = "strong-phrase"
        elif count_matches(self.re_weak, head) and count_matches(self.re_nouns, head):
            decision = "weak-verb+software-noun"
        if decision:
            return 1, f"decision context ({decision})"

        tech_context = bool(count_matches(self.re_nouns, head)) or bool(provider_hit)
        factor_hit = any(count_matches(p, head) for _lbl, p in self.factors.values())
        if tech_context and (factor_hit or provider_hit):
            return 2, "discourse (factor/provider without decision phrasing)"
        return 0, "legal domain but no tech/factor context"

    # -- sentiment helpers -----------------------------------------------------
    def sentiment_of(self, text):
        if not self.score or not text.strip():
            return None
        return round(self.score(text), 4)

    def label_of(self, value):
        if value is None:
            return ""
        if value >= self.pos_thr:
            return "positive"
        if value <= self.neg_thr:
            return "negative"
        return "neutral"

    # -- per-thread measurement --------------------------------------------------
    def measure(self, op_text, comment_texts):
        full_text = " \n ".join([op_text] + comment_texts)
        sents = sentences_of(full_text)

        build_hits = count_matches(self.re_build, full_text)
        buy_hits = count_matches(self.re_buy, full_text)
        if build_hits >= self.stance_min and build_hits >= self.stance_ratio * max(buy_hits, 1):
            stance = "build-leaning"
        elif buy_hits >= self.stance_min and buy_hits >= self.stance_ratio * max(build_hits, 1):
            stance = "buy-leaning"
        elif build_hits >= 2 and buy_hits >= 2:
            stance = "mixed"
        else:
            stance = "unclear"

        factor_rows = {}
        for key, (label, pattern) in self.factors.items():
            mentions = count_matches(pattern, full_text)
            if not mentions:
                continue
            tones = [self.sentiment_of(s) for s in sents if count_matches(pattern, s)]
            tones = [t for t in tones if t is not None]
            tone = round(sum(tones) / len(tones), 4) if tones else None
            factor_rows[key] = (label, mentions, tone)

        provider_rows = {}
        for name, category, legal_specific, pattern in self.providers:
            mentions = count_matches(pattern, full_text)
            if not mentions:
                continue
            tones = [self.sentiment_of(s) for s in sents if count_matches(pattern, s)]
            tones = [t for t in tones if t is not None]
            tone = round(sum(tones) / len(tones), 4) if tones else None
            provider_rows[name] = (category, mentions, tone)

        op_sent = self.sentiment_of(op_text)
        comment_sents = [self.sentiment_of(c) for c in comment_texts]
        comment_sents = [c for c in comment_sents if c is not None]
        comments_mean = round(sum(comment_sents) / len(comment_sents), 4) if comment_sents else None

        return {
            "stance": stance, "build_hits": build_hits, "buy_hits": buy_hits,
            "factors": factor_rows, "providers": provider_rows,
            "op_sentiment": op_sent, "op_label": self.label_of(op_sent),
            "comments_sentiment": comments_mean, "comments_label": self.label_of(comments_mean),
        }

    # -- step-B discovery: stemmed vocabulary ------------------------------------
    def stem_counts(self, full_text):
        """Return (stem Counter, {stem: surface-form Counter}) for one thread."""
        counts = Counter()
        surfaces = {}
        for tok in tokenize(clean_text(full_text)):
            if len(tok) < self.min_len or tok in self.stopwords:
                continue
            stem = self.stem(tok)
            counts[stem] += 1
            surfaces.setdefault(stem, Counter())[tok] += 1
        return counts, surfaces

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


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def load_wrapper_texts(wrapper):
    """Extract (title, selftext, comment_texts, meta) from a cached listing."""
    listing = wrapper["listing"]
    post = listing[0]["data"]["children"][0]["data"]
    title = post.get("title", "")
    selftext = post.get("selftext", "")
    if selftext in ("[deleted]", "[removed]"):
        selftext = ""
    comments = [clean_text(c) for c in walk_comments(listing[1]["data"]["children"])]
    meta = {
        "created": datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
        .strftime("%Y-%m-%d") if post.get("created_utc") else "",
        "score": post.get("score", ""),
        "num_comments": post.get("num_comments", ""),
    }
    return clean_text(title), clean_text(selftext), comments, meta


def run(args):
    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)

    analyzer = Analyzer(config, args.sentiment)
    os.makedirs(args.out, exist_ok=True)

    subreddits = ([s.strip() for s in args.subreddits.split(",") if s.strip()]
                  if args.subreddits else
                  config["subreddits"]["specialist"] + config["subreddits"]["practitioner"])
    queries = ([q.strip() for q in args.queries.split(";") if q.strip()]
               if args.queries else config["queries"])

    print("🚀 Build-vs-Buy pipeline v2.1")
    print(f"   sentiment backend : {analyzer.sentiment_backend}")
    print(f"   stemmer           : {analyzer.stem_backend}")
    print(f"   subreddits        : {len(subreddits)} | queries: {len(queries)}")

    stats = {}
    wrappers = []

    if args.reanalyze:
        thread_dir = os.path.join(args.cache, "threads")
        files = sorted(os.listdir(thread_dir)) if os.path.isdir(thread_dir) else []
        print(f"♻️  Re-analysing {len(files)} cached threads (no network).")
        for fname in files:
            with open(os.path.join(thread_dir, fname), encoding="utf-8") as fh:
                wrappers.append(json.load(fh))
    else:
        client = RedditClient(args.cache, args.sleep_thread, args.sleep_sub,
                              args.user_agent)
        seen = set()
        for sub in subreddits:
            stats[sub] = {"results": 0, "tier1": 0, "tier2": 0, "fetched": 0}
            print(f"🔍 r/{sub}")
            for query in queries:
                for post in client.search(sub, query, args.limit, args.pages):
                    thread_id = post.get("id")
                    if not thread_id or thread_id in seen:
                        continue
                    seen.add(thread_id)
                    stats[sub]["results"] += 1
                    # Pre-filter on the search result itself: saves one full
                    # thread download per off-topic result (rate-limit budget).
                    tier, _ = analyzer.tier(sub, post.get("title", ""),
                                            post.get("selftext", ""))
                    if tier == 0:
                        continue
                    stats[sub][f"tier{tier}"] += 1
                    wrapper = client.fetch_thread(sub, thread_id,
                                                  post.get("permalink", ""))
                    if wrapper:
                        stats[sub]["fetched"] += 1
                        wrappers.append(wrapper)
                time.sleep(client.sleep_thread)
            time.sleep(client.sleep_sub)
            print(f"   -> {stats[sub]['results']} unique results | "
                  f"tier1 {stats[sub]['tier1']} | tier2 {stats[sub]['tier2']} | "
                  f"downloaded {stats[sub]['fetched']}")

    # ---- per-thread analysis --------------------------------------------------
    master_rows = []
    keyword_rows_raw = []   # (sub, url, stem, count) -> display form filled later
    factor_agg = {}         # key -> {"label", tiers: {1: {...}, 2: {...}}}
    provider_agg = {}       # name -> {"category", "threads", "t1_threads", "mentions", "tones"}
    stem_total = Counter()
    stem_threads = Counter()
    stem_surfaces = {}

    for wrapper in wrappers:
        sub = wrapper["subreddit"]
        title, selftext, comments, meta = load_wrapper_texts(wrapper)
        tier, reason = analyzer.tier(sub, title, selftext)
        if tier == 0:
            continue
        op_text = f"{title}\n{selftext}".strip()
        result = analyzer.measure(op_text, comments)

        master_rows.append({
            "subreddit": sub,
            "thread_id": wrapper["thread_id"],
            "thread_url": wrapper["thread_url"],
            "title": title,
            "created_utc": meta["created"],
            "post_score": meta["score"],
            "num_comments_reported": meta["num_comments"],
            "n_comments_scraped": len(comments),
            "corpus_tier": tier,
            "tier_reason": reason,
            "stance_heuristic": result["stance"],
            "build_signal_hits": result["build_hits"],
            "buy_signal_hits": result["buy_hits"],
            "providers_mentioned": "; ".join(
                f"{n} ({m[1]})" for n, m in sorted(result["providers"].items())),
            "factors_mentioned": "; ".join(
                f"{lbl} ({m})" for _k, (lbl, m, _t) in sorted(result["factors"].items())),
            "op_sentiment": result["op_sentiment"],
            "op_sentiment_label": result["op_label"],
            "comments_sentiment_mean": result["comments_sentiment"],
            "comments_sentiment_label": result["comments_label"],
            "manual_decision": "",
            "manual_outcome": "",
            "manual_notes": "",
        })

        for key, (label, mentions, tone) in result["factors"].items():
            agg = factor_agg.setdefault(key, {"label": label, "tiers": {}})
            bucket = agg["tiers"].setdefault(tier, {"threads": 0, "mentions": 0, "tones": []})
            bucket["threads"] += 1
            bucket["mentions"] += mentions
            if tone is not None:
                bucket["tones"].append(tone)

        for name, (category, mentions, tone) in result["providers"].items():
            agg = provider_agg.setdefault(name, {"category": category, "threads": 0,
                                                 "t1_threads": 0, "mentions": 0, "tones": []})
            agg["threads"] += 1
            agg["t1_threads"] += 1 if tier == 1 else 0
            agg["mentions"] += mentions
            if tone is not None:
                agg["tones"].append(tone)

        # step-B discovery accumulation (whole corpus, both tiers)
        full_text = " \n ".join([op_text] + comments)
        counts, surfaces = analyzer.stem_counts(full_text)
        for stem, count in counts.items():
            stem_total[stem] += count
            stem_threads[stem] += 1
            stem_surfaces.setdefault(stem, Counter()).update(surfaces[stem])
        if not args.no_legacy:
            for stem, count in counts.items():
                if count >= analyzer.min_count:
                    keyword_rows_raw.append((sub, wrapper["thread_url"], stem, count))

    n_threads = len(master_rows)
    n_tier1 = sum(1 for r in master_rows if r["corpus_tier"] == 1)
    n_tier2 = n_threads - n_tier1
    print(f"\n📊 {n_threads} threads kept ({n_tier1} decision-tier, {n_tier2} discourse-tier).")

    # ---- write outputs ---------------------------------------------------------
    def write_csv(name, fieldnames, rows):
        path = os.path.join(args.out, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"   💾 {path} ({len(rows)} rows)")

    if master_rows:
        write_csv("threads_master.csv", list(master_rows[0].keys()), master_rows)

    def tier_cells(bucket, base):
        if not bucket:
            return 0, 0, ""
        tones = bucket["tones"]
        tone = round(sum(tones) / len(tones), 4) if tones else ""
        pct = round(100 * bucket["threads"] / base, 1) if base else 0
        return bucket["threads"], pct, tone

    factor_rows = []
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
    write_csv("factor_salience.csv",
              ["factor", "label", "n_threads_decision", "pct_decision_threads",
               "tone_decision", "n_threads_discourse", "pct_discourse_threads",
               "tone_discourse", "n_threads_all", "pct_all_threads",
               "total_mentions", "tone_all"], factor_rows)

    provider_rows = []
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
    write_csv("provider_mentions.csv",
              ["provider", "category", "n_threads", "n_threads_decision",
               "pct_of_threads", "total_mentions", "mean_sentence_sentiment"],
              provider_rows)

    # step-B discovery table: review 'uncovered' terms before freezing factors
    display = {stem: surfaces.most_common(1)[0][0]
               for stem, surfaces in stem_surfaces.items()}
    discovery_rows = []
    for stem, total in stem_total.most_common():
        if total < analyzer.min_count:
            continue
        top_surfaces = [s for s, _ in stem_surfaces[stem].most_common(3)]
        discovery_rows.append({
            "stem": stem,
            "display_form": display[stem],
            "surface_forms": "; ".join(top_surfaces),
            "total_count": total,
            "n_threads": stem_threads[stem],
            "pct_of_threads": round(100 * stem_threads[stem] / n_threads, 1) if n_threads else 0,
            "covered_by": analyzer.coverage_of(top_surfaces),
        })
    write_csv("term_discovery.csv",
              ["stem", "display_form", "surface_forms", "total_count",
               "n_threads", "pct_of_threads", "covered_by"], discovery_rows)

    if not args.no_legacy:
        keyword_rows = [{"subreddit": s, "thread_url": u,
                         "term": display.get(stem, stem), "stem": stem, "count": c}
                        for s, u, stem, c in sorted(keyword_rows_raw,
                                                    key=lambda r: (r[0], r[1], -r[3]))]
        write_csv("keyword_frequency.csv",
                  ["subreddit", "thread_url", "term", "stem", "count"], keyword_rows)

    # ---- run summary (for the Methodology / Results sections) ------------------
    summary_path = os.path.join(args.out, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Run finished: {datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"Sentiment backend: {analyzer.sentiment_backend}\n")
        fh.write(f"Stemmer: {analyzer.stem_backend}\n")
        fh.write(f"Queries: {queries}\n")
        fh.write(f"Subreddits: {subreddits}\n")
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
        for row in factor_rows[:12]:
            fh.write(f"  {row['label']}: {row['n_threads_decision']} decision threads "
                     f"({row['pct_decision_threads']}%), tone {row['tone_decision']} "
                     f"| all: {row['n_threads_all']} ({row['pct_all_threads']}%)\n")
        fh.write("\nTop providers by thread coverage:\n")
        for row in provider_rows[:15]:
            fh.write(f"  {row['provider']} [{row['category']}]: {row['n_threads']} threads, "
                     f"{row['total_mentions']} mentions, tone {row['mean_sentence_sentiment']}\n")
        fh.write("\nTop UNCOVERED candidate terms (factor-gap check, step B):\n")
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
    print("🎉 Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/keywords.json")
    parser.add_argument("--out", default="out")
    parser.add_argument("--cache", default="cache",
                        help="raw thread JSON cache (enables offline --reanalyze)")
    parser.add_argument("--reanalyze", action="store_true",
                        help="skip scraping; rebuild all outputs from cached threads")
    parser.add_argument("--limit", type=int, default=100,
                        help="search results per query per subreddit (max 100)")
    parser.add_argument("--pages", type=int, default=1,
                        help="search pages per query (each page = up to --limit results)")
    parser.add_argument("--sleep-thread", type=float, default=3.5)
    parser.add_argument("--sleep-sub", type=float, default=6.0)
    parser.add_argument("--sentiment", choices=["auto", "vader", "textblob", "none"],
                        default="auto")
    parser.add_argument("--no-legacy", action="store_true",
                        help="skip the per-thread keyword_frequency.csv output")
    parser.add_argument("--subreddits", default="",
                        help="comma-separated override of the config subreddit list")
    parser.add_argument("--queries", default="",
                        help="semicolon-separated override of the config query list")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
