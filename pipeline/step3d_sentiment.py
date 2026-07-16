"""STEP 3d — SENTIMENT (OPTIONAL): VADER sentiment per kept thread.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3d_sentiment.json

The entire sentiment method lives in THIS file only, so it can be excluded
from the thesis without touching anything else:
  - skip it for one run:        python3 run_pipeline.py --skip-sentiment
  - or delete this file:        the orchestrator notes it and carries on
Either way every other table is still produced; only the sentiment/tone
columns stay empty.

Measured per thread (Hutto & Gilbert 2014; thresholds in config "sentiment"):
  - compound score of the original post   (author-satisfaction proxy)
  - mean compound over the comments       (community reaction)
  - mean tone of sentences mentioning each factor / provider (the dictionaries
    are compiled from config, NOT read from the other modules' outputs, so
    this module stays independent of them)

Standalone use:
  python3 -m pipeline.step3d_sentiment --config config/keywords.json --out out
"""

import argparse
import os

from . import common
from .common import build_sentiment, count_matches, sentences_of


class SentimentMeasurer:
    def __init__(self, config, sentiment_choice="auto"):
        self.score, self.backend = build_sentiment(sentiment_choice)
        senti = config["sentiment"]
        self.pos_thr = senti["positive_threshold"]
        self.neg_thr = senti["negative_threshold"]
        self.factors = common.compile_factor_patterns(config)
        self.providers = common.compile_provider_patterns(config)

    def sentiment_of(self, text):
        if not self.score or not text.strip():
            return None
        return round(self.score(text), 4)

    def label_of(self, value):
        if value is None:
            return ""
        if value >= self.pos_thr:
            return "positive"
        if value <= self.neg_thr:
            return "negative"
        return "neutral"

    def _mean_tone(self, pattern, sents):
        tones = [self.sentiment_of(s) for s in sents if count_matches(pattern, s)]
        tones = [t for t in tones if t is not None]
        return round(sum(tones) / len(tones), 4) if tones else None

    def measure(self, op_text, comment_texts):
        full_text = " \n ".join([op_text] + comment_texts)
        sents = sentences_of(full_text)

        factor_tones = {}
        for key, (_label, pattern) in self.factors.items():
            tone = self._mean_tone(pattern, sents)
            if tone is not None:
                factor_tones[key] = tone

        provider_tones = {}
        for name, _category, _legal_specific, pattern in self.providers:
            tone = self._mean_tone(pattern, sents)
            if tone is not None:
                provider_tones[name] = tone

        op_sent = self.sentiment_of(op_text)
        comment_sents = [self.sentiment_of(c) for c in comment_texts]
        comment_sents = [c for c in comment_sents if c is not None]
        comments_mean = round(sum(comment_sents) / len(comment_sents), 4) if comment_sents else None

        return {
            "op_sentiment": op_sent, "op_label": self.label_of(op_sent),
            "comments_sentiment": comments_mean,
            "comments_label": self.label_of(comments_mean),
            "factor_tones": factor_tones, "provider_tones": provider_tones,
        }


def run(config_path, out_dir, sentiment_choice="auto"):
    measurer = SentimentMeasurer(common.load_config(config_path), sentiment_choice)
    pairs = common.kept_thread_wrappers(out_dir)
    print(f"💬 Step 3d — sentiment for {len(pairs)} kept threads "
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
    parser.add_argument("--sentiment", choices=["auto", "vader", "textblob", "none"],
                        default="auto")
    args = parser.parse_args()
    run(args.config, args.out, args.sentiment)


if __name__ == "__main__":
    main()
