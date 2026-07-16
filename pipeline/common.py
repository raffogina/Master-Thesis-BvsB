"""Shared building blocks used by every pipeline step.

Nothing in this module does network access or writes pipeline outputs; it only
provides pure helpers (text cleaning, term matching, stemming, sentiment,
cache-file reading) so that the step modules stay small and independent.

Method references: VADER sentiment (Hutto & Gilbert 2014); dictionary-based
content analysis (Krippendorff 2018; Grimmer & Stewart 2013); Porter stemming
(Porter 1980). No usernames are collected or stored anywhere in the pipeline.
"""

import csv
import html
import json
import os
import random
import re
import time
from datetime import datetime, timezone

DEFAULT_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

COLLECTOR_NAME = "arctic-shift archive (research mirror + local query screening)"

# All intermediate step outputs live in this sub-folder of --out; the final
# thesis tables stay at the top level of --out, exactly as in scraper_v2.
STEPS_DIRNAME = "steps"

MANIFEST_NAME = "step1_collection_manifest.csv"
RUN_PARAMS_NAME = "step1_run_params.json"
TIERS_NAME = "step2_thread_tiers.csv"
STANCE_NAME = "step3a_stance.json"
FACTORS_NAME = "step3b_factors.json"
PROVIDERS_NAME = "step3c_providers.json"
SENTIMENT_NAME = "step3d_sentiment.json"
TERM_COUNTS_NAME = "step3e_term_counts.json"


def steps_dir(out_dir):
    return os.path.join(out_dir, STEPS_DIRNAME)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def human_sleep(base):
    """Sleep around `base` seconds with human-like variation.

    Uniform jitter (0.7x-1.7x) breaks the fixed request beat, and a small
    share of calls adds a longer 'reading' pause so the overall timing
    pattern stays irregular rather than metronomic.
    """
    delay = base * random.uniform(0.7, 1.7)
    if random.random() < 0.07:
        delay += random.uniform(15, 40)
    time.sleep(delay)


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


def compile_factor_patterns(config):
    """{factor_key: (label, compiled pattern)} from config['factors']."""
    factors = {}
    for key, spec in config["factors"].items():
        if key.startswith("_"):
            continue
        factors[key] = (spec["label"], compile_terms(spec["terms"]))
    return factors


def compile_provider_patterns(config):
    """[(name, category, legal_specific, compiled pattern)] from config['providers']."""
    providers = []
    for entry in config["providers"]["entries"]:
        providers.append((entry["name"], entry["category"],
                          bool(entry.get("legal_specific")),
                          compile_terms(entry["patterns"])))
    return providers


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
# Cached-thread reading (the cache in cache/threads/*.json is the only data
# store shared by steps 1-3; wrappers are never modified after collection)
# ---------------------------------------------------------------------------

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


def load_wrapper(cache_path):
    with open(cache_path, encoding="utf-8") as fh:
        return json.load(fh)


def load_wrapper_texts(wrapper):
    """Extract (title, selftext, comment_texts, meta) from a cached wrapper.

    Two cache formats are supported so re-analysis works across collectors:
      - "listing": the raw Reddit JSON listing (reddit-json collector)
      - "post":    pre-extracted texts written by non-JSON collectors
    Returns None when the wrapper holds no post (deleted/empty thread).
    """
    if "listing" in wrapper:
        listing = wrapper["listing"]
        children = listing[0].get("data", {}).get("children", [])
        if not children:
            return None
        post = children[0]["data"]
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
    else:
        post = wrapper.get("post")
        if not post:
            return None
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        if selftext in ("[deleted]", "[removed]"):
            selftext = ""
        comments = [clean_text(c) for c in post.get("comments", [])
                    if c and c not in ("[deleted]", "[removed]")]
        meta = {
            "created": post.get("created", ""),
            "score": post.get("score", ""),
            "num_comments": post.get("num_comments", ""),
        }
    return clean_text(title), clean_text(selftext), comments, meta


def extract_op_and_comments(wrapper):
    """(op_text, comment_texts) for one cached wrapper, or None if empty.

    Every step-3 analysis module MUST build its texts through this helper so
    all of them measure exactly the same character stream.
    """
    extracted = load_wrapper_texts(wrapper)
    if extracted is None:
        return None
    title, selftext, comments, _meta = extracted
    return f"{title}\n{selftext}".strip(), comments


def kept_thread_wrappers(out_dir):
    """[(tiers_row, wrapper)] for every thread step 2 kept (tier 1 or 2).

    Shared by all step-3 analysis modules: each re-reads the cache through
    this one function, so the modules stay mutually independent (no module
    consumes another module's output) while still iterating the identical
    thread list in the identical order.
    """
    sdir = steps_dir(out_dir)
    kept = [r for r in read_csv(os.path.join(sdir, TIERS_NAME))
            if r["corpus_tier"] in ("1", "2")]
    cache_file_of = {r["thread_id"]: r["cache_file"]
                     for r in read_csv(os.path.join(sdir, MANIFEST_NAME))}
    pairs = []
    for row in kept:
        cache_file = cache_file_of.get(row["thread_id"])
        if not cache_file or not os.path.exists(cache_file):
            print(f"   ⚠️ cache file missing for thread {row['thread_id']} — skipped")
            continue
        pairs.append((row, load_wrapper(cache_file)))
    return pairs


# ---------------------------------------------------------------------------
# Small I/O helpers shared by the step modules
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_subreddits(config, override_csv=""):
    if override_csv:
        return [s.strip() for s in override_csv.split(",") if s.strip()]
    return config["subreddits"]["specialist"] + config["subreddits"]["practitioner"]


def resolve_queries(config, override_csv=""):
    if override_csv:
        return [q.strip() for q in override_csv.split(";") if q.strip()]
    return config["queries"]


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"   💾 {path} ({len(rows)} rows)")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print(f"   💾 {path}")


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
