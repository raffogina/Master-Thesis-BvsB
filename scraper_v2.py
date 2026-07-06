#!/usr/bin/env python3
"""
Build-vs-Buy Reddit evidence pipeline (v2)
Master Thesis: "The 'Build vs. Buy' Dilemma in Legal Departments"

Pipeline:
  1. COLLECT   search each subreddit with the queries in config/keywords.json via
               Reddit's public JSON endpoints (rate-limited, retried, cached to disk
               so analysis can be re-run offline without re-scraping).
  2. GATE      keep a thread only if title+body show explicit decision context AND
               legal-domain evidence (single documented rule; see config "gate").
  3. MEASURE   per gated thread: decision-stance heuristic (build/buy leaning),
               decision-factor mentions, provider mentions, and VADER sentiment
               (original post = author-satisfaction proxy; comments = community
               reaction; sentence-level tone per factor and provider).
  4. AGGREGATE factor salience table, provider barometer, per-thread master table
               with empty manual-annotation columns, and an optional cleaned/
               lemmatised keyword-frequency table (replacement of the v1 output).

Outputs (in --out, default ./out):
  threads_master.csv     one row per gated thread
  factor_salience.csv    factor x coverage x tone
  provider_mentions.csv  provider x coverage x tone
  keyword_frequency.csv  lemmatised inductive word counts (disable with --no-legacy)
  run_summary.txt        parameters + per-subreddit yield (for the Methodology section)

Method references: VADER sentiment (Hutto & Gilbert 2014); dictionary-based content
analysis (Krippendorff 2018; Grimmer & Stewart 2013); Reddit research practice and
ethics (Proferes et al. 2021). No usernames are collected or stored.
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

DEFAULT_USER_AGENT = "python:build-vs-buy-thesis-scraper:v2.0 (academic research; MLB thesis)"

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
# Lemmatisation (folds plural/singular so 'firm'/'firms' count together)
# ---------------------------------------------------------------------------

def build_lemmatizer():
    """Prefer NLTK WordNet; fall back to conservative suffix rules."""
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        try:
            lem.lemmatize("firms")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            lem.lemmatize("firms")
        return lem.lemmatize, "nltk-wordnet"
    except Exception:
        def fold(word):
            if len(word) > 4 and word.endswith("ies"):
                return word[:-3] + "y"
            if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
                return word[:-2]
            if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
                return word[:-1]
            return word
        return fold, "suffix-rules (install nltk for WordNet lemmatisation)"


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
        self.lemmatize, self.lemma_backend = build_lemmatizer()

    # -- gating ------------------------------------------------------------
    def gate(self, subreddit, title, selftext):
        """Return (passes: bool, reason: str). Rule documented in config."""
        head = clean_text(f"{title}\n{selftext}")
        decision = None
        if count_matches(self.re_strong, head):
            decision = "strong-phrase"
        elif count_matches(self.re_weak, head) and count_matches(self.re_nouns, head):
            decision = "weak-verb+software-noun"
        if not decision:
            return False, "no decision context"
        if count_matches(self.re_legal, head):
            return True, f"{decision} & legal-term"
        if subreddit.lower() in self.specialist_subs:
            return True, f"{decision} & specialist-subreddit"
        for name, _cat, legal_specific, pattern in self.providers:
            if legal_specific and count_matches(pattern, head):
                return True, f"{decision} & legal-provider ({name})"
        return False, "decision context but no legal-domain evidence"

    # -- sentiment helpers ---------------------------------------------------
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

    # -- per-thread measurement ----------------------------------------------
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

    # -- legacy inductive keyword counts --------------------------------------
    def keyword_counts(self, full_text):
        counts = Counter()
        for tok in tokenize(clean_text(full_text)):
            if len(tok) < self.min_len or tok in self.stopwords:
                continue
            lemma = self.lemmatize(tok)
            if len(lemma) < self.min_len or lemma in self.stopwords:
                continue
            counts[lemma] += 1
        return {w: c for w, c in counts.items() if c >= self.min_count}


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

    print("🚀 Build-vs-Buy pipeline v2")
    print(f"   sentiment backend : {analyzer.sentiment_backend}")
    print(f"   lemmatiser        : {analyzer.lemma_backend}")
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
            stats[sub] = {"results": 0, "gated_in": 0, "fetched": 0}
            print(f"🔍 r/{sub}")
            for query in queries:
                for post in client.search(sub, query, args.limit, args.pages):
                    thread_id = post.get("id")
                    if not thread_id or thread_id in seen:
                        continue
                    seen.add(thread_id)
                    stats[sub]["results"] += 1
                    # Pre-gate on the search result itself: saves one full
                    # thread download per off-topic result (rate-limit budget).
                    passes, _ = analyzer.gate(sub, post.get("title", ""),
                                              post.get("selftext", ""))
                    if not passes:
                        continue
                    stats[sub]["gated_in"] += 1
                    wrapper = client.fetch_thread(sub, thread_id,
                                                  post.get("permalink", ""))
                    if wrapper:
                        stats[sub]["fetched"] += 1
                        wrappers.append(wrapper)
                time.sleep(client.sleep_thread)
            time.sleep(client.sleep_sub)
            print(f"   -> {stats[sub]['results']} unique results, "
                  f"{stats[sub]['gated_in']} passed gate, "
                  f"{stats[sub]['fetched']} downloaded")

    # ---- per-thread analysis ------------------------------------------------
    master_rows = []
    keyword_rows = []
    factor_agg = {}    # key -> dict(threads, mentions, tones[])
    provider_agg = {}  # name -> dict(category, threads, mentions, tones[])

    for wrapper in wrappers:
        sub = wrapper["subreddit"]
        title, selftext, comments, meta = load_wrapper_texts(wrapper)
        passes, reason = analyzer.gate(sub, title, selftext)
        if not passes:
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
            "gate_reason": reason,
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
            agg = factor_agg.setdefault(key, {"label": label, "threads": 0,
                                              "mentions": 0, "tones": []})
            agg["threads"] += 1
            agg["mentions"] += mentions
            if tone is not None:
                agg["tones"].append(tone)

        for name, (category, mentions, tone) in result["providers"].items():
            agg = provider_agg.setdefault(name, {"category": category, "threads": 0,
                                                 "mentions": 0, "tones": []})
            agg["threads"] += 1
            agg["mentions"] += mentions
            if tone is not None:
                agg["tones"].append(tone)

        if not args.no_legacy:
            full_text = " \n ".join([op_text] + comments)
            for lemma, count in sorted(analyzer.keyword_counts(full_text).items(),
                                       key=lambda kv: -kv[1]):
                keyword_rows.append({"subreddit": sub,
                                     "thread_url": wrapper["thread_url"],
                                     "lemma": lemma, "count": count})

    n_threads = len(master_rows)
    print(f"\n📊 {n_threads} threads passed the gate and were analysed.")

    # ---- write outputs -------------------------------------------------------
    def write_csv(name, fieldnames, rows):
        path = os.path.join(args.out, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"   💾 {path} ({len(rows)} rows)")

    if master_rows:
        write_csv("threads_master.csv", list(master_rows[0].keys()), master_rows)

    factor_rows = []
    for key, agg in sorted(factor_agg.items(), key=lambda kv: -kv[1]["threads"]):
        tones = agg["tones"]
        factor_rows.append({
            "factor": key,
            "label": agg["label"],
            "n_threads": agg["threads"],
            "pct_of_threads": round(100 * agg["threads"] / n_threads, 1) if n_threads else 0,
            "total_mentions": agg["mentions"],
            "mean_sentence_sentiment": round(sum(tones) / len(tones), 4) if tones else "",
        })
    write_csv("factor_salience.csv",
              ["factor", "label", "n_threads", "pct_of_threads",
               "total_mentions", "mean_sentence_sentiment"], factor_rows)

    provider_rows = []
    for name, agg in sorted(provider_agg.items(),
                            key=lambda kv: (-kv[1]["threads"], -kv[1]["mentions"])):
        tones = agg["tones"]
        provider_rows.append({
            "provider": name,
            "category": agg["category"],
            "n_threads": agg["threads"],
            "pct_of_threads": round(100 * agg["threads"] / n_threads, 1) if n_threads else 0,
            "total_mentions": agg["mentions"],
            "mean_sentence_sentiment": round(sum(tones) / len(tones), 4) if tones else "",
        })
    write_csv("provider_mentions.csv",
              ["provider", "category", "n_threads", "pct_of_threads",
               "total_mentions", "mean_sentence_sentiment"], provider_rows)

    if not args.no_legacy:
        write_csv("keyword_frequency.csv",
                  ["subreddit", "thread_url", "lemma", "count"], keyword_rows)

    # ---- run summary (for the Methodology / Results sections) ---------------
    summary_path = os.path.join(args.out, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Run finished: {datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"Sentiment backend: {analyzer.sentiment_backend}\n")
        fh.write(f"Lemmatiser: {analyzer.lemma_backend}\n")
        fh.write(f"Queries: {queries}\n")
        fh.write(f"Subreddits: {subreddits}\n")
        fh.write(f"Threads analysed (post-gate): {n_threads}\n")
        dates = sorted(r["created_utc"] for r in master_rows if r["created_utc"])
        if dates:
            fh.write(f"Thread date range: {dates[0]} to {dates[-1]}\n")
        if stats:
            fh.write("\nPer-subreddit yield (unique results / gate passed / downloaded):\n")
            for sub, s in stats.items():
                fh.write(f"  r/{sub}: {s['results']} / {s['gated_in']} / {s['fetched']}\n")
        fh.write("\nTop factors by thread coverage:\n")
        for row in factor_rows[:10]:
            fh.write(f"  {row['label']}: {row['n_threads']} threads "
                     f"({row['pct_of_threads']}%), tone {row['mean_sentence_sentiment']}\n")
        fh.write("\nTop providers by thread coverage:\n")
        for row in provider_rows[:15]:
            fh.write(f"  {row['provider']} [{row['category']}]: {row['n_threads']} threads, "
                     f"{row['total_mentions']} mentions, tone {row['mean_sentence_sentiment']}\n")
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
                        help="skip the inductive keyword_frequency.csv output")
    parser.add_argument("--subreddits", default="",
                        help="comma-separated override of the config subreddit list")
    parser.add_argument("--queries", default="",
                        help="semicolon-separated override of the config query list")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
