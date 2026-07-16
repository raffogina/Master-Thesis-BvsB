"""STEP 3b — FACTORS: decision-factor mentions per kept thread.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3b_factors.json

Counts each config factor dictionary (dictionary-based content analysis:
Krippendorff 2018; Grimmer & Stewart 2013) over the full thread text
(OP + all comments). Mentions only — the OPTIONAL sentence tone per factor
comes from the separate sentiment module (step3d) and is joined at aggregate
time, so factor salience works with sentiment excluded from the thesis.

Standalone use:
  python3 -m pipeline.step3b_factors --config config/keywords.json --out out
"""

import argparse
import os

from . import common
from .common import count_matches


class FactorMeasurer:
    def __init__(self, config):
        self.factors = common.compile_factor_patterns(config)

    def measure(self, op_text, comment_texts):
        """{factor_key: [label, mentions]} for factors with >= 1 mention."""
        full_text = " \n ".join([op_text] + comment_texts)
        rows = {}
        for key, (label, pattern) in self.factors.items():
            mentions = count_matches(pattern, full_text)
            if mentions:
                rows[key] = [label, mentions]
        return rows


def run(config_path, out_dir):
    measurer = FactorMeasurer(common.load_config(config_path))
    pairs = common.kept_thread_wrappers(out_dir)
    print(f"🧩 Step 3b — factor mentions for {len(pairs)} kept threads")

    records = []
    for row, wrapper in pairs:
        texts = common.extract_op_and_comments(wrapper)
        if texts is None:
            continue
        op_text, comments = texts
        records.append({"thread_id": row["thread_id"],
                        "factors": measurer.measure(op_text, comments)})

    path = os.path.join(common.steps_dir(out_dir), common.FACTORS_NAME)
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
