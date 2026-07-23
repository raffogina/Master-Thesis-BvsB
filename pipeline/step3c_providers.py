"""STEP 3c — PROVIDERS: legal-tech provider mentions per kept thread.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3c_providers.json

Counts each config provider pattern over the full thread text (OP + all
comments) — the vendor-landscape barometer. Mentions only.

Standalone use:
  python3 -m pipeline.step3c_providers --config config/keywords.json --out out
"""

import argparse
import os

from . import common


class ProviderMeasurer:
    def __init__(self, config):
        self.providers = common.compile_provider_patterns(config)

    def measure(self, op_text, comment_texts):
        """{provider_name: [category, mentions]} for providers with >= 1 mention."""
        full_text = " \n ".join([op_text] + comment_texts)
        rows = {}
        for entry in self.providers:
            name, category = entry[0], entry[1]
            mentions = common.count_provider_mentions(entry, full_text)
            if mentions:
                rows[name] = [category, mentions]
        return rows


def run(config_path, out_dir):
    measurer = ProviderMeasurer(common.load_config(config_path))
    pairs = common.kept_thread_wrappers(out_dir)
    print(f"🏢 Step 3c — provider mentions for {len(pairs)} kept threads")

    records = []
    for row, wrapper in pairs:
        texts = common.extract_op_and_comments(wrapper)
        if texts is None:
            continue
        op_text, comments = texts
        records.append({"thread_id": row["thread_id"],
                        "providers": measurer.measure(op_text, comments)})

    path = os.path.join(common.steps_dir(out_dir), common.PROVIDERS_NAME)
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
