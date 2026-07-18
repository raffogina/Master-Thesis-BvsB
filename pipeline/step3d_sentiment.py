"""STEP 3d — SENTIMENT: mean tone of first-person past-decision sentences.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3d_sentiment.json

The entire sentiment method lives in THIS file only, and it measures exactly
ONE thing: for every TIER-1 (decision) thread, the mean VADER compound score
(Hutto & Gilbert 2014) of the sentences that report a decision actually taken
in the past. A sentence qualifies only when it contains a direct
subject-verb bigram: a first-person subject (config sentiment.decision_
subjects: "i"/"we") followed by a decision verb (config
sentiment.decision_verbs) with at most one tolerated adverb in between
(common.PHRASE_GAP_ADVERBS: just/recently/finally - the same single-adverb
tolerance every other dictionary term in this pipeline gets), e.g. "we
built", "I decided", "we just bought", "we finally migrated". Matching is
done sentence by sentence (split on . ! ?), never across the whole
post/thread, so "I work at a firm... they migrated to X last year" does not
count as "I migrated" just because both appear in the same document - and
within one sentence, "I heard they migrated to X" still does not match,
because no "i migrated"/"we migrated" bigram (with at most one adverb
between subject and verb) is present (the word right before "migrated" is
"they", not "i"/"we").
Both dictionaries live in config/keywords.json like every other instrument
(dictionary-based content analysis: Krippendorff 2018; Grimmer & Stewart 2013).
No other sentiment is measured anywhere in the pipeline.

Standalone use:
  python3 -m pipeline.step3d_sentiment --config config/keywords.json --out out
"""

import argparse
import os
import re

from . import common
from .common import PHRASE_GAP_ADVERBS, build_sentiment, sentences_of


class SentimentMeasurer:
    def __init__(self, config):
        self.score, self.backend = build_sentiment()
        senti = config["sentiment"]
        subjects = "|".join(re.escape(s) for s in senti["decision_subjects"])
        verbs = "|".join(re.escape(v) for v in senti["decision_verbs"])
        adverbs = "|".join(re.escape(a) for a in PHRASE_GAP_ADVERBS)
        self.re_subject_verb = re.compile(
            rf"\b(?:{subjects})(?:\s+(?:{adverbs}))?\s+(?:{verbs})\b", re.IGNORECASE)

    def decision_sentences(self, op_text, comment_texts):
        """Sentences (OP + comments) containing a direct subject-verb bigram
        ('we built', 'i decided', ...) - see module docstring."""
        full_text = " \n ".join([op_text] + comment_texts)
        return [s for s in sentences_of(full_text) if self.re_subject_verb.search(s)]

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
