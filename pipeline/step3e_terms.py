"""STEP 3e — TERMS (OPTIONAL): Porter-stemmed token counts per kept thread.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3e_term_counts.json

Feeds the step-B factor-gap cross-check (term_discovery.csv). Every word
variant is folded onto its stem (Porter 1980) and each stem remembers its
surface forms so it can be displayed through the most frequent one. Skip with
--skip-terms (or delete this file): the discovery table is then not produced,
everything else still is.

Standalone use:
  python3 -m pipeline.step3e_terms --config config/keywords.json --out out
"""

import argparse
import os
from collections import Counter

from . import common
from .common import BASE_STOPWORDS, build_stemmer, clean_text, tokenize


class TermCounter:
    def __init__(self, config):
        terms_cfg = config["terms"]
        self.stopwords = BASE_STOPWORDS | set(terms_cfg["extra_stopwords"])
        self.min_len = terms_cfg["min_token_len"]
        self.stem, self.stem_backend = build_stemmer()

    def count(self, full_text):
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


def run(config_path, out_dir):
    counter = TermCounter(common.load_config(config_path))
    pairs = common.kept_thread_wrappers(out_dir)
    print(f"🔤 Step 3e — term counts for {len(pairs)} kept threads "
          f"(stemmer: {counter.stem_backend})")

    records = []
    for row, wrapper in pairs:
        texts = common.extract_op_and_comments(wrapper)
        if texts is None:
            continue
        op_text, comments = texts
        full_text = " \n ".join([op_text] + comments)
        counts, surfaces = counter.count(full_text)
        records.append({
            "thread_id": row["thread_id"],
            "subreddit": wrapper["subreddit"],
            "thread_url": wrapper["thread_url"],
            "counts": dict(counts),                                  # {stem: count}
            "surfaces": {s: dict(c) for s, c in surfaces.items()},   # {stem: {form: count}}
        })

    path = os.path.join(common.steps_dir(out_dir), common.TERM_COUNTS_NAME)
    common.write_json(path, {"stemmer": counter.stem_backend,
                             "measured_at": common.utc_now_iso(),
                             "threads": records})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/keywords.json")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
