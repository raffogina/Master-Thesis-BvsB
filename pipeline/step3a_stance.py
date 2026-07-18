"""STEP 3a — STANCE: build/buy-leaning heuristic per kept thread.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3a_stance.json

Counts the config stance.build_signals / stance.buy_signals dictionaries over
the full thread text (OP + all comments) and classifies the discussion as
build-leaning / buy-leaning / mixed / unclear using the config thresholds
(min_hits, dominance_ratio). This is a heuristic ORIENTATION, not a verified
decision — validate a sample by hand (README validation protocol).

Threads with fewer than stance.min_comments scraped comments are classified
'too_short' instead of 'unclear': they simply don't have enough text for the
hit counts to reach min_hits, which is a different phenomenon from a thread
that discusses build and buy in genuinely ambiguous proportions.

Independent of every other step-3 module: no sentiment, no factor/provider
dictionaries. Deleting or breaking any other analysis module does not affect
this output, and vice versa.

Standalone use:
  python3 -m pipeline.step3a_stance --config config/keywords.json --out out
"""

import argparse
import os

from . import common
from .common import compile_terms, count_matches


class StanceMeasurer:
    def __init__(self, config):
        stance = config["stance"]
        self.re_build = compile_terms(stance["build_signals"])
        self.re_buy = compile_terms(stance["buy_signals"])
        self.stance_min = stance.get("min_hits", 3)
        self.stance_ratio = stance.get("dominance_ratio", 2.0)
        self.min_comments = stance.get("min_comments", 0)

    def measure(self, op_text, comment_texts):
        full_text = " \n ".join([op_text] + comment_texts)
        build_hits = count_matches(self.re_build, full_text)
        buy_hits = count_matches(self.re_buy, full_text)
        if len(comment_texts) < self.min_comments:
            stance = "too_short"
        elif build_hits >= self.stance_min and build_hits >= self.stance_ratio * max(buy_hits, 1):
            stance = "build-leaning"
        elif buy_hits >= self.stance_min and buy_hits >= self.stance_ratio * max(build_hits, 1):
            stance = "buy-leaning"
        elif build_hits >= 2 and buy_hits >= 2:
            stance = "mixed"
        else:
            stance = "unclear"
        return {"stance": stance, "build_hits": build_hits, "buy_hits": buy_hits}


def run(config_path, out_dir):
    measurer = StanceMeasurer(common.load_config(config_path))
    pairs = common.kept_thread_wrappers(out_dir)
    print(f"⚖️  Step 3a — stance heuristic for {len(pairs)} kept threads")

    records = []
    for row, wrapper in pairs:
        texts = common.extract_op_and_comments(wrapper)
        if texts is None:
            continue
        op_text, comments = texts
        result = measurer.measure(op_text, comments)
        result["thread_id"] = row["thread_id"]
        records.append(result)

    path = os.path.join(common.steps_dir(out_dir), common.STANCE_NAME)
    common.write_json(path, {"measured_at": common.utc_now_iso(), "threads": records})
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
