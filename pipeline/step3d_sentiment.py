"""STEP 3d — SENTIMENT: mean tone of first-person past-decision sentences.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3d_sentiment.json

The entire sentiment method lives in THIS file only, and it measures exactly
ONE thing: for every TIER-1 (decision) thread, the mean VADER compound score
(Hutto & Gilbert 2014) of the sentences that report a decision actually taken
in the past. A sentence qualifies only when it contains BOTH
  - a past-decision marker  (config sentiment.past_decision_markers), and
  - a first-person marker   (config sentiment.first_person_markers),
so first-hand experiences ("we bought Clio and regret it") are measured while
hypotheticals and advice to third parties ("you should buy X") are not.
Both dictionaries live in config/keywords.json like every other instrument
(dictionary-based content analysis: Krippendorff 2018; Grimmer & Stewart 2013).
No other sentiment is measured anywhere in the pipeline.

Standalone use:
  python3 -m pipeline.step3d_sentiment --config config/keywords.json --out out
"""

import argparse
import os

from . import common
from .common import build_sentiment, compile_terms, count_matches, sentences_of


class SentimentMeasurer:
    def __init__(self, config):
        self.score, self.backend = build_sentiment()
        senti = config["sentiment"]
        self.re_decision = compile_terms(senti["past_decision_markers"])
        self.re_first_person = compile_terms(senti["first_person_markers"])

    def decision_sentences(self, op_text, comment_texts):
        """Sentences (OP + comments) with a past-decision AND a first-person
        marker."""
        full_text = " \n ".join([op_text] + comment_texts)
        return [s for s in sentences_of(full_text)
                if count_matches(self.re_decision, s)
                and count_matches(self.re_first_person, s)]

    def measure(self, op_text, comment_texts):
        tones = [self.score(s)
                 for s in self.decision_sentences(op_text, comment_texts)]
        mean = round(sum(tones) / len(tones), 4) if tones else None
        return {"decision_tone_mean": mean, "n_decision_sentences": len(tones)}


def run(config_path, out_dir):
    measurer = SentimentMeasurer(common.load_config(config_path))
    pairs = [(row, wrapper) for row, wrapper in common.kept_thread_wrappers(out_dir)
             if row["corpus_tier"] == "1"]
    print(f"💬 Step 3d — decision-sentence tone for {len(pairs)} tier-1 threads "
          f"(backend: {measurer.backend})")

    records = []
    for row, wrapper in pairs:
        texts = common.extract_op_and_comments(wrapper)
        if texts is None:
            continue
        op_text, comments = texts
        result = measurer.measure(op_text, comments)
        result["thread_id"] = row["thread_id"]
        records.append(result)

    path = os.path.join(common.steps_dir(out_dir), common.SENTIMENT_NAME)
    common.write_json(path, {"sentiment_backend": measurer.backend,
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
