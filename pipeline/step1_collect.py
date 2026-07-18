"""STEP 1 — COLLECT: gather candidate threads into the local cache.

NO network access anywhere in this module. Two modes:
  local (default)      read every *.jsonl file in --data-dir (default ./data)
                       — the subreddit dumps downloaded from the Arctic Shift
                       research archive (archive approach per Pushshift,
                       Baumgartner et al. 2020). File names do not matter:
                       records are told apart by content (posts have a
                       'title', comments have a 'body' + 'link_id') and
                       comments are attached to their post via link_id across
                       files. Every post found is screened against the study
                       queries and the tier rule; matching threads are
                       (re)built in cache/threads/ — the dumps are
                       authoritative, so an existing cache file for the same
                       thread is overwritten.
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
from collections import Counter
from datetime import datetime, timezone

from . import common
from .common import clean_text, count_matches, compile_terms
from .step2_tier import TierAssigner

COLLECTOR_LOCAL = ("arctic-shift subreddit dumps (local *.jsonl files, "
                   "full local query screening, no network)")

MANIFEST_FIELDS = ["thread_id", "subreddit", "thread_url", "cache_file"]


def _query_pattern(query):
    """Translate a study query ('a OR b', '"x y" OR z') into one locally
    applied regex, reusing the dictionary term rules (word bounds,
    space==hyphen)."""
    terms = [part.strip().strip('"').strip()
             for part in query.split(" OR ")]
    return compile_terms([t for t in terms if t])


def _collect_offline(cache_dir):
    """--reanalyze: inventory every cached thread, no network. Sorted by file
    name so re-runs are deterministic."""
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
                         f"(see data/README.md), or use --reanalyze.")
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.lower().endswith(".jsonl"))
    if not files:
        raise SystemExit(f"🛑 No .jsonl files in {data_dir}\n"
                         f"   Save the Arctic Shift subreddit dumps there "
                         f"(see data/README.md), or use --reanalyze.")
    return files


def _read_dump_posts(files):
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


def _collect_from_dumps(config, cache_dir, data_dir, queries):
    """Local-dump collection (no network). Every subreddit found in the dump
    files is analysed — subreddit selection is done upstream by curating which
    dumps are saved into --data-dir, not by this function. Returns
    (manifest_rows, stats, total posts screened)."""
    assigner = TierAssigner(config)   # pre-filter == step-2 rule
    files = _dump_files(data_dir)
    print(f"   📂 reading {len(files)} .jsonl file(s) from {data_dir}")
    posts, counters = _read_dump_posts(files)
    print(f"   📚 {len(posts)} unique posts and {counters['comment_lines']} comment "
          f"lines found"
          + (f" | {counters['bad_lines']} unreadable line(s) skipped"
             if counters["bad_lines"] else "")
          + (f" | {counters['duplicate_posts']} duplicate post(s) ignored"
             if counters["duplicate_posts"] else ""))
    if not posts:
        raise SystemExit("🛑 The .jsonl files contain no posts — check that you "
                         "downloaded the *posts* dumps, not only comments.")

    ordered_subs = sorted({p["subreddit"] for p in posts.values()})

    patterns = [_query_pattern(q) for q in queries]

    # screen every post: any study query + the step-2 tier rule on title+body
    kept_posts, stats = [], {}
    for sub in ordered_subs:
        sub_posts = sorted((p for p in posts.values() if p["subreddit"] == sub),
                           key=lambda p: -int(float(p.get("created_utc", 0) or 0)))
        stats[sub] = {"posts_total": len(sub_posts), "results": 0,
                      "tier1": 0, "tier2": 0, "fetched": 0}
        for post in sub_posts:
            text = clean_text(f"{post['title']}\n{post['selftext']}")
            if patterns and not any(count_matches(pat, text) for pat in patterns):
                continue
            stats[sub]["results"] += 1
            tier, _ = assigner.tier(post["title"], post["selftext"])
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
        # entry for the same thread
        cache_path = os.path.join(cache_dir, "threads", f"{tid}.json")
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(wrapper, fh)
        stats[sub]["fetched"] += 1
        manifest_rows.append({"thread_id": tid, "subreddit": sub,
                              "thread_url": wrapper["thread_url"],
                              "cache_file": cache_path})

    for sub in ordered_subs:
        s = stats[sub]
        print(f"   🔍 r/{sub}: {s['posts_total']} posts | {s['results']} query matches | "
              f"tier1 {s['tier1']} | tier2 {s['tier2']} | threads built {s['fetched']}")
    return manifest_rows, stats, len(posts)


def run(config_path, out_dir, cache_dir, reanalyze=False, args=None):
    """Execute step 1 and write manifest + run-params files."""
    config = common.load_config(config_path)
    queries = common.resolve_queries(config, args.queries if args else "")
    data_dir = getattr(args, "data_dir", "data") if args else "data"

    if reanalyze:
        mode, collector = "offline-reanalyze", common.COLLECTOR_NAME
        print("📥 Step 1 — collect (offline cache inventory)")
        manifest_rows, stats = _collect_offline(cache_dir)
        posts_total = None   # unknown without the dumps
    else:
        mode, collector = "local-dumps", COLLECTOR_LOCAL
        print("📥 Step 1 — collect (local .jsonl dumps, no network)")
        # every subreddit found in the folder is analysed; subreddit
        # selection happens upstream, by choosing which dumps go in --data-dir
        manifest_rows, stats, posts_total = _collect_from_dumps(
            config, cache_dir, data_dir, queries)

    subreddits = sorted({row["subreddit"] for row in manifest_rows})

    manifest_path = os.path.join(common.steps_dir(out_dir), common.MANIFEST_NAME)
    common.write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)

    params_path = os.path.join(common.steps_dir(out_dir), common.RUN_PARAMS_NAME)
    common.write_json(params_path, {
        "mode": mode,
        "collector": collector,
        "collected_at": common.utc_now_iso(),
        "queries": queries,
        "subreddits": subreddits,
        "posts_total": posts_total,
        "stats": stats,
    })
    return manifest_path


def add_cli_arguments(parser):
    """Collection flags, shared with the orchestrator CLI."""
    parser.add_argument("--config", default="config/keywords.json")
    parser.add_argument("--out", default="out")
    parser.add_argument("--cache", default="cache",
                        help="raw thread JSON cache (enables offline --reanalyze)")
    parser.add_argument("--data-dir", default="data",
                        help="folder with the Arctic Shift .jsonl dumps "
                             "(every *.jsonl file in it is read; names don't matter)")
    parser.add_argument("--reanalyze", action="store_true",
                        help="skip collection; inventory cached threads instead")
    parser.add_argument("--queries", default="",
                        help="semicolon-separated override of the config query list")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_arguments(parser)
    args = parser.parse_args()
    run(args.config, args.out, args.cache, reanalyze=args.reanalyze, args=args)


if __name__ == "__main__":
    main()
