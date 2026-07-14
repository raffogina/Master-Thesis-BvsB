"""STEP 2 — TIER: assign every collected thread to a corpus tier.

Input   out/steps/step1_collection_manifest.csv  (+ the cached thread files)
Output  out/steps/step2_thread_tiers.csv         one row per manifest thread

Tier rule (documented in config "gate", applied to title+body only so the
pre-download filter in step 1 and this step use the identical criterion):
  tier  1  explicit decision context AND legal-domain evidence
  tier  2  no decision phrasing, but legal-domain evidence, a tech context,
           and at least one study factor or provider
  tier  0  excluded (off-topic noise)
  tier -1  cached wrapper unreadable/empty (deleted thread) — excluded

Tier 0 / -1 threads are kept in this file with their exclusion reason (audit
trail for the Methodology section) but are ignored by steps 3 and 4.

Standalone use:
  python3 -m pipeline.step2_tier --config config/keywords.json --out out
"""

import argparse
import os

from . import common
from .common import clean_text, compile_terms, count_matches

TIERS_FIELDS = ["thread_id", "subreddit", "thread_url", "title", "created_utc",
                "post_score", "num_comments_reported", "n_comments_scraped",
                "corpus_tier", "tier_reason"]


class TierAssigner:
    """Compiles the gate dictionaries and applies the corpus-tier rule."""

    def __init__(self, config):
        gate = config["gate"]
        self.re_strong = compile_terms(gate["strong_phrases"])
        self.re_weak = compile_terms(gate["weak_verbs"])
        self.re_nouns = compile_terms(gate["software_nouns"])
        self.re_legal = compile_terms(gate["legal_terms"])
        self.specialist_subs = {s.lower() for s in config["subreddits"]["specialist"]}
        self.factors = common.compile_factor_patterns(config)
        self.providers = common.compile_provider_patterns(config)

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


def run(config_path, out_dir):
    """Read the step-1 manifest, tier every thread, write the tiers table."""
    config = common.load_config(config_path)
    assigner = TierAssigner(config)

    manifest_path = os.path.join(common.steps_dir(out_dir), common.MANIFEST_NAME)
    manifest = common.read_csv(manifest_path)
    print(f"🏷️  Step 2 — tiering {len(manifest)} collected threads")

    rows = []
    counts = {1: 0, 2: 0, 0: 0, -1: 0}
    for entry in manifest:
        wrapper = common.load_wrapper(entry["cache_file"])
        extracted = common.load_wrapper_texts(wrapper)
        if extracted is None:
            counts[-1] += 1
            rows.append({
                "thread_id": entry["thread_id"], "subreddit": entry["subreddit"],
                "thread_url": entry["thread_url"], "title": "", "created_utc": "",
                "post_score": "", "num_comments_reported": "", "n_comments_scraped": 0,
                "corpus_tier": -1, "tier_reason": "unreadable or empty wrapper",
            })
            continue
        title, selftext, comments, meta = extracted
        tier, reason = assigner.tier(entry["subreddit"], title, selftext)
        counts[tier] += 1
        rows.append({
            "thread_id": entry["thread_id"], "subreddit": entry["subreddit"],
            "thread_url": entry["thread_url"], "title": title,
            "created_utc": meta["created"], "post_score": meta["score"],
            "num_comments_reported": meta["num_comments"],
            "n_comments_scraped": len(comments),
            "corpus_tier": tier, "tier_reason": reason,
        })

    tiers_path = os.path.join(common.steps_dir(out_dir), common.TIERS_NAME)
    common.write_csv(tiers_path, TIERS_FIELDS, rows)
    print(f"   -> tier1 {counts[1]} | tier2 {counts[2]} | "
          f"excluded {counts[0]} | unreadable {counts[-1]}")
    return tiers_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/keywords.json")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
