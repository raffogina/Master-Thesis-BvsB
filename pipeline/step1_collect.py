"""STEP 1 — COLLECT: gather candidate threads into the local cache.

Three modes:
  local (default)      NO network, NO api key: read every *.jsonl file in
                       --data-dir (default ./data) — the subreddit dumps
                       downloaded from the Arctic Shift research archive
                       (archive approach per Pushshift, Baumgartner et al.
                       2020). File names do not matter: records are told
                       apart by content (posts have a 'title', comments have
                       a 'body' + 'link_id') and comments are attached to
                       their post via link_id across files. Every post found
                       is screened against the study queries and the tier
                       rule; matching threads are (re)built in cache/threads/
                       — the dumps are authoritative, so an existing cache
                       file for the same thread is overwritten.
  --source archive-api the previous online collector, kept for the audit
                       trail: list the newest --pages x 100 threads per
                       subreddit from the Arctic Shift API, screen locally,
                       download matches into the cache (never overwrites).
  --reanalyze          inventory the threads already in the cache so steps
                       2-4 can be re-run without any data source at all.

Outputs (everything later steps need — they never read the data source):
  cache/threads/<id>.json                      raw thread wrappers
  out/steps/step1_collection_manifest.csv      one row per thread to analyse
  out/steps/step1_run_params.json              parameters + per-subreddit yield

Posts that already fail the corpus-tier rule on their title+body are not
turned into threads; the rule is imported from step 2 so both steps apply the
identical criterion (it is re-applied in step 2 to the cached text as the
authoritative assignment).

Standalone use:
  python3 -m pipeline.step1_collect --config config/keywords.json --out out
  python3 -m pipeline.step1_collect --data-dir data      # local dumps (default)
  python3 -m pipeline.step1_collect --reanalyze          # cache inventory
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import requests

from . import common
from .common import clean_text, count_matches, compile_terms, human_sleep
from .step2_tier import TierAssigner

ARCTIC_SHIFT = "https://arctic-shift.photon-reddit.com"

COLLECTOR_LOCAL = ("arctic-shift subreddit dumps (local *.jsonl files, "
                   "full local query screening, no network)")

MANIFEST_FIELDS = ["thread_id", "subreddit", "thread_url", "cache_file"]


class ArchiveClient:
    """Collects from the Arctic Shift research archive of Reddit.

    Arctic Shift (github.com/ArthurHeitmann/arctic_shift) ingests Reddit
    continuously (typical lag: hours, verified) and serves posts/comments in
    Reddit's own field schema, so the downstream pipeline is unchanged.

    Its full-text search endpoint is intermittently under maintenance, so
    collection instead lists the newest --pages x 100 posts per subreddit
    once and screens them locally against each study query (title+selftext,
    same wildcard/phrase rules as the dictionaries). Comments are fetched
    by link_id, capped at MAX_COMMENTS per thread.
    """

    MAX_COMMENTS = 500  # same practical cap as a rendered Reddit thread page

    def __init__(self, cache_dir, sleep_thread, sleep_sub, user_agent):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.cache_dir = cache_dir
        self.sleep_thread = sleep_thread
        self.sleep_sub = sleep_sub
        self._post_by_id = {}
        self._listing_cache = {}
        os.makedirs(os.path.join(cache_dir, "threads"), exist_ok=True)

    def _get_json(self, url, params=None):
        """GET with retry on rate limits (429/422), maintenance (503), drops."""
        for attempt in range(5):
            try:
                resp = self.session.get(url, params=params, timeout=60)
            except requests.exceptions.RequestException:
                wait = 10 * (attempt + 1)
                print(f"   🚨 Network drop. Retrying in {wait}s ...")
                time.sleep(wait)
                continue
            if resp.status_code in (429, 422):  # 422 = archive's slow-down signal
                print("   ⚠️ Archive asks to slow down. Cooling 35s ...")
                time.sleep(35)
                continue
            if resp.status_code == 503:
                print("   ⚠️ Archive busy/maintenance (503). Retrying in 60s ...")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                return None, resp.status_code
            try:
                return resp.json(), 200
            except ValueError:
                return None, -1
        return None, -2

    @staticmethod
    def _query_pattern(query):
        """Translate a Reddit-search query ('a OR b', '"x y" OR z') into one
        locally applied regex, reusing the dictionary term rules (word
        bounds, space==hyphen)."""
        terms = [part.strip().strip('"').strip()
                 for part in query.split(" OR ")]
        return compile_terms([t for t in terms if t])

    def _list_posts(self, subreddit, pages):
        """Fetch (once per subreddit) the newest pages x 100 posts."""
        if subreddit in self._listing_cache:
            return self._listing_cache[subreddit]
        posts, before = [], None
        for _ in range(pages):
            params = {"subreddit": subreddit, "limit": 100, "sort": "desc"}
            if before is not None:
                params["before"] = before
            data, status = self._get_json(f"{ARCTIC_SHIFT}/api/posts/search",
                                          params)
            if data is None:
                print(f"   ⚠️ Listing stopped for r/{subreddit} (status {status})")
                break
            batch = data.get("data", [])
            posts.extend(batch)
            if len(batch) < 100:
                break
            before = int(float(batch[-1].get("created_utc", 0))) or None
            if before is None:
                break
            human_sleep(self.sleep_thread)
        self._listing_cache[subreddit] = posts
        if posts:
            oldest = datetime.fromtimestamp(
                int(float(posts[-1].get("created_utc", 0))),
                tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"   📚 screening {len(posts)} archived threads "
                  f"(back to {oldest})")
        return posts

    def search(self, subreddit, query, limit, pages):
        """Yield archived submissions whose title+selftext match the query.

        Full selftext is available (not a snippet), so the pre-download tier
        filter sees the same text as the final analysis. Matches are kept in
        memory so fetch_thread() does not need to re-request them.
        """
        pattern = self._query_pattern(query)
        matched = 0
        for post in self._list_posts(subreddit, pages):
            if matched >= limit * pages:
                return
            text = clean_text(f"{post.get('title', '')}\n{post.get('selftext', '')}")
            if pattern and not count_matches(pattern, text):
                continue
            if post.get("id"):
                self._post_by_id[post["id"]] = post
                matched += 1
                yield post

    def fetch_thread(self, subreddit, thread_id, permalink):
        """Fetch (or load from cache) comments and build a text wrapper."""
        cache_path = os.path.join(self.cache_dir, "threads", f"{thread_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                return json.load(fh)
        post = self._post_by_id.get(thread_id, {})

        comments, before = [], None
        while len(comments) < self.MAX_COMMENTS:
            human_sleep(self.sleep_thread)
            params = {"link_id": f"t3_{thread_id}", "limit": 100, "sort": "desc"}
            if before is not None:
                params["before"] = before
            data, status = self._get_json(f"{ARCTIC_SHIFT}/api/comments/search",
                                          params)
            if data is None:
                print(f"   ⚠️ Comments skipped for {thread_id} (status {status})")
                break
            batch = data.get("data", [])
            for comment in batch:
                body = comment.get("body", "")
                if body and body not in ("[deleted]", "[removed]"):
                    comments.append(body)
            if len(batch) < 100:
                break
            before = int(float(batch[-1].get("created_utc", 0))) or None
            if before is None:
                break

        created_utc = post.get("created_utc")
        selftext = post.get("selftext", "")
        wrapper = {
            "subreddit": subreddit,
            "thread_id": thread_id,
            "thread_url": f"https://www.reddit.com{permalink}",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "arctic-shift-archive",
            "post": {
                "title": post.get("title", ""),
                "selftext": "" if selftext in ("[deleted]", "[removed]") else selftext,
                "comments": comments,
                "created": (datetime.fromtimestamp(int(float(created_utc)),
                                                   tz=timezone.utc).strftime("%Y-%m-%d")
                            if created_utc else ""),
                "score": post.get("score", ""),
                "num_comments": post.get("num_comments", ""),
            },
        }
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(wrapper, fh)
        return wrapper


def _collect_live(config, args, subreddits, queries):
    """Network collection. Returns (manifest_rows, per-subreddit stats)."""
    assigner = TierAssigner(config)   # pre-download filter == step-2 rule
    client = ArchiveClient(args.cache, args.sleep_thread, args.sleep_sub,
                           args.user_agent)
    manifest_rows, stats, seen = [], {}, set()
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
                tier, _ = assigner.tier(sub, post.get("title", ""),
                                        post.get("selftext", ""))
                if tier == 0:
                    continue
                stats[sub][f"tier{tier}"] += 1
                wrapper = client.fetch_thread(sub, thread_id,
                                              post.get("permalink", ""))
                if wrapper:
                    stats[sub]["fetched"] += 1
                    manifest_rows.append({
                        "thread_id": thread_id,
                        "subreddit": sub,
                        "thread_url": wrapper["thread_url"],
                        "cache_file": os.path.join(args.cache, "threads",
                                                   f"{thread_id}.json"),
                    })
            human_sleep(client.sleep_thread)
        human_sleep(client.sleep_sub)
        print(f"   -> {stats[sub]['results']} unique results | "
              f"tier1 {stats[sub]['tier1']} | tier2 {stats[sub]['tier2']} | "
              f"downloaded {stats[sub]['fetched']}")
    return manifest_rows, stats


def _collect_offline(cache_dir):
    """--reanalyze: inventory every cached thread, no network. Sorted by file
    name so re-runs are deterministic (same order as scraper_v2 --reanalyze)."""
    thread_dir = os.path.join(cache_dir, "threads")
    files = (sorted(f for f in os.listdir(thread_dir) if f.endswith(".json"))
             if os.path.isdir(thread_dir) else [])
    print(f"♻️  Offline mode: inventorying {len(files)} cached threads (no network).")
    manifest_rows = []
    for fname in files:
        path = os.path.join(thread_dir, fname)
        wrapper = common.load_wrapper(path)
        manifest_rows.append({
            "thread_id": wrapper.get("thread_id", fname[:-5]),
            "subreddit": wrapper.get("subreddit", ""),
            "thread_url": wrapper.get("thread_url", ""),
            "cache_file": path,
        })
    return manifest_rows, {}


def _dump_files(data_dir):
    """Every *.jsonl file in the data folder, sorted for deterministic runs."""
    if not os.path.isdir(data_dir):
        raise SystemExit(f"🛑 Data folder not found: {data_dir}\n"
                         f"   Create it and save the Arctic Shift .jsonl dumps there "
                         f"(see data/README.md), or use --reanalyze / --source archive-api.")
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.lower().endswith(".jsonl"))
    if not files:
        raise SystemExit(f"🛑 No .jsonl files in {data_dir}\n"
                         f"   Save the Arctic Shift subreddit dumps there "
                         f"(see data/README.md), or use --reanalyze / --source archive-api.")
    return files


def _read_dump_posts(files, sub_filter):
    """Pass 1: every submission in the dumps (slim fields only).

    Records are identified by CONTENT, not by file name: a submission has a
    'title'; a comment has a 'body' + 'link_id'. Returns ({id: post}, counters).
    """
    posts, counters = {}, Counter()
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    counters["bad_lines"] += 1
                    continue
                if not isinstance(rec, dict):
                    counters["bad_lines"] += 1
                    continue
                if "title" in rec and rec.get("id"):
                    sub = rec.get("subreddit", "")
                    if sub_filter is not None and sub.lower() not in sub_filter:
                        counters["posts_filtered_out"] += 1
                        continue
                    if rec["id"] in posts:
                        counters["duplicate_posts"] += 1
                        continue
                    posts[rec["id"]] = {
                        "id": rec["id"],
                        "subreddit": sub,
                        "title": rec.get("title", ""),
                        "selftext": rec.get("selftext", "") or "",
                        "permalink": rec.get("permalink", ""),
                        "created_utc": rec.get("created_utc", 0),
                        "score": rec.get("score", ""),
                        "num_comments": rec.get("num_comments", ""),
                    }
                elif "body" in rec:
                    counters["comment_lines"] += 1
                else:
                    counters["other_records"] += 1
    return posts, counters


def _read_dump_comments(files, kept_ids):
    """Pass 2: comment bodies for the kept threads only, attached via link_id
    and sorted chronologically. Deleted/removed comments are dropped (same
    rule as the online collector)."""
    by_thread = {tid: [] for tid in kept_ids}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict) or "body" not in rec or "title" in rec:
                    continue
                link = rec.get("link_id") or ""
                tid = link[3:] if link.startswith("t3_") else link
                if tid not in by_thread:
                    continue
                body = rec.get("body", "")
                if body and body not in ("[deleted]", "[removed]"):
                    try:
                        created = int(float(rec.get("created_utc", 0) or 0))
                    except (TypeError, ValueError):
                        created = 0
                    by_thread[tid].append((created, body))
    return {tid: [body for _created, body in sorted(pairs, key=lambda p: p[0])]
            for tid, pairs in by_thread.items()}


def _collect_from_dumps(config, cache_dir, data_dir, queries, sub_filter):
    """Local-dump collection (no network). Returns (manifest_rows, stats,
    subreddits found)."""
    assigner = TierAssigner(config)   # pre-filter == step-2 rule
    files = _dump_files(data_dir)
    print(f"   📂 reading {len(files)} .jsonl file(s) from {data_dir}")
    posts, counters = _read_dump_posts(files, sub_filter)
    print(f"   📚 {len(posts)} unique posts and {counters['comment_lines']} comment "
          f"lines found"
          + (f" | {counters['bad_lines']} unreadable line(s) skipped"
             if counters["bad_lines"] else "")
          + (f" | {counters['duplicate_posts']} duplicate post(s) ignored"
             if counters["duplicate_posts"] else ""))
    if not posts:
        raise SystemExit("🛑 The .jsonl files contain no posts — check that you "
                         "downloaded the *posts* dumps, not only comments.")

    # deterministic subreddit order: config order first, then extras A-Z
    found_subs = {p["subreddit"] for p in posts.values()}
    config_order = (config["subreddits"]["specialist"]
                    + config["subreddits"]["practitioner"])
    ordered_subs = ([s for s in config_order if s in found_subs]
                    + sorted(found_subs - set(config_order)))
    extras = sorted(found_subs - set(config_order))
    if extras:
        print(f"   ℹ️ subreddit(s) not in the config lists, analysed anyway: "
              f"{', '.join(extras)}")

    patterns = [ArchiveClient._query_pattern(q) for q in queries]

    # screen every post: any study query + the step-2 tier rule on title+body
    kept_posts, stats = [], {}
    for sub in ordered_subs:
        stats[sub] = {"results": 0, "tier1": 0, "tier2": 0, "fetched": 0}
        sub_posts = sorted((p for p in posts.values() if p["subreddit"] == sub),
                           key=lambda p: -int(float(p.get("created_utc", 0) or 0)))
        for post in sub_posts:
            text = clean_text(f"{post['title']}\n{post['selftext']}")
            if patterns and not any(count_matches(pat, text) for pat in patterns):
                continue
            stats[sub]["results"] += 1
            tier, _ = assigner.tier(sub, post["title"], post["selftext"])
            if tier == 0:
                continue
            stats[sub][f"tier{tier}"] += 1
            kept_posts.append(post)

    comments_of = _read_dump_comments(files, {p["id"] for p in kept_posts})

    os.makedirs(os.path.join(cache_dir, "threads"), exist_ok=True)
    manifest_rows = []
    for post in kept_posts:
        sub, tid = post["subreddit"], post["id"]
        selftext = post["selftext"]
        created_utc = post.get("created_utc")
        permalink = post.get("permalink") or f"/r/{sub}/comments/{tid}/"
        wrapper = {
            "subreddit": sub,
            "thread_id": tid,
            "thread_url": f"https://www.reddit.com{permalink}",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "arctic-shift-local-dump",
            "post": {
                "title": post["title"],
                "selftext": "" if selftext in ("[deleted]", "[removed]") else selftext,
                "comments": comments_of.get(tid, []),
                "created": (datetime.fromtimestamp(int(float(created_utc)),
                                                   tz=timezone.utc).strftime("%Y-%m-%d")
                            if created_utc else ""),
                "score": post.get("score", ""),
                "num_comments": post.get("num_comments", ""),
            },
        }
        # the dumps are the authoritative source: overwrite any older cache
        # entry for the same thread (e.g. one fetched by the api collector)
        cache_path = os.path.join(cache_dir, "threads", f"{tid}.json")
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(wrapper, fh)
        stats[sub]["fetched"] += 1
        manifest_rows.append({"thread_id": tid, "subreddit": sub,
                              "thread_url": wrapper["thread_url"],
                              "cache_file": cache_path})

    for sub in ordered_subs:
        s = stats[sub]
        print(f"   🔍 r/{sub}: {s['results']} query matches | tier1 {s['tier1']} | "
              f"tier2 {s['tier2']} | threads built {s['fetched']}")
    return manifest_rows, stats, ordered_subs


def run(config_path, out_dir, cache_dir, reanalyze=False, args=None):
    """Execute step 1 and write manifest + run-params files."""
    config = common.load_config(config_path)
    subreddits = common.resolve_subreddits(config, args.subreddits if args else "")
    queries = common.resolve_queries(config, args.queries if args else "")
    source = getattr(args, "source", "local") if args else "local"
    data_dir = getattr(args, "data_dir", "data") if args else "data"

    if reanalyze:
        mode, collector = "offline-reanalyze", common.COLLECTOR_NAME
        print("📥 Step 1 — collect (offline cache inventory)")
        manifest_rows, stats = _collect_offline(cache_dir)
    elif source == "archive-api":
        mode, collector = "archive-api", common.COLLECTOR_NAME
        print("📥 Step 1 — collect (live archive api)")
        manifest_rows, stats = _collect_live(config, args, subreddits, queries)
    else:
        mode, collector = "local-dumps", COLLECTOR_LOCAL
        print("📥 Step 1 — collect (local .jsonl dumps, no network)")
        # default: analyse EVERY subreddit found in the folder; --subreddits
        # narrows the run to the listed ones
        sub_filter = ({s.lower() for s in subreddits}
                      if args is not None and args.subreddits else None)
        manifest_rows, stats, subreddits = _collect_from_dumps(
            config, cache_dir, data_dir, queries, sub_filter)

    manifest_path = os.path.join(common.steps_dir(out_dir), common.MANIFEST_NAME)
    common.write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)

    params_path = os.path.join(common.steps_dir(out_dir), common.RUN_PARAMS_NAME)
    common.write_json(params_path, {
        "mode": mode,
        "collector": collector,
        "collected_at": common.utc_now_iso(),
        "queries": queries,
        "subreddits": list(subreddits),
        "stats": stats,
    })
    return manifest_path


def add_cli_arguments(parser):
    """Collection flags, shared with the orchestrator CLI."""
    parser.add_argument("--config", default="config/keywords.json")
    parser.add_argument("--out", default="out")
    parser.add_argument("--cache", default="cache",
                        help="raw thread JSON cache (enables offline --reanalyze)")
    parser.add_argument("--source", choices=["local", "archive-api"], default="local",
                        help="data source: 'local' = the .jsonl dumps in --data-dir "
                             "(default, no network); 'archive-api' = the previous "
                             "online Arctic Shift collector")
    parser.add_argument("--data-dir", default="data",
                        help="folder with the Arctic Shift .jsonl dumps "
                             "(every *.jsonl file in it is read; names don't matter)")
    parser.add_argument("--reanalyze", action="store_true",
                        help="skip collection; inventory cached threads instead")
    parser.add_argument("--limit", type=int, default=100,
                        help="[archive-api only] cap on locally matched threads per "
                             "query (effective cap is limit x pages)")
    parser.add_argument("--pages", type=int, default=5,
                        help="[archive-api only] x100 newest archived posts screened "
                             "per subreddit")
    parser.add_argument("--sleep-thread", type=float, default=3.5,
                        help="[archive-api only]")
    parser.add_argument("--sleep-sub", type=float, default=6.0,
                        help="[archive-api only]")
    parser.add_argument("--subreddits", default="",
                        help="comma-separated subreddit restriction (local mode "
                             "default: every subreddit found in the data folder)")
    parser.add_argument("--queries", default="",
                        help="semicolon-separated override of the config query list")
    parser.add_argument("--user-agent", default=common.DEFAULT_USER_AGENT,
                        help="[archive-api only]")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_arguments(parser)
    args = parser.parse_args()
    run(args.config, args.out, args.cache, reanalyze=args.reanalyze, args=args)


if __name__ == "__main__":
    main()
