"""STEP 3d — SENTIMENT: tone of first-person past-decision sentences.

Input   out/steps/step2_thread_tiers.csv (+ manifest + cached threads)
Output  out/steps/step3d_sentiment.json

The entire sentiment method lives in THIS file only. For every TIER-1
(decision) thread it finds every sentence that reports a decision actually
taken in the past, and scores EACH ONE separately with VADER (Hutto &
Gilbert 2014) - nothing is collapsed into a single number, since a thread
with several commenters can hold several decisions with different tone and
even different build/buy direction.

A sentence qualifies as a decision sentence only when BOTH hold:
  1. it contains a direct subject-verb bigram: a first-person subject
     (config sentiment.decision_subjects: "i"/"we") followed by a decision
     verb (config sentiment.decision_verbs) with at most one tolerated
     adverb in between (common.PHRASE_GAP_ADVERBS: just/recently/finally -
     the same single-adverb tolerance every other dictionary term in this
     pipeline gets), e.g. "we built", "I decided", "we just bought", "we
     finally migrated".
  2. the SAME sentence also names a legaltech term (config
     gate.legaltech_terms) or a recognized provider (config
     providers.entries) - many decision_verbs double as everyday, non-
     software phrasing ("we dropped the case", "we picked a jury", "we
     settled on a court date") that has nothing to do with a procurement
     decision; see config sentiment._ambiguity_rule.

Matching is done sentence by sentence (split on . ! ?), never across the
whole post/thread, so "I work at a firm... they migrated to X last year"
does not count as "I migrated" just because both appear in the same
document - and within one sentence, "I heard they migrated to X" still does
not match, because no "i migrated"/"we migrated" bigram (with at most one
adverb between subject and verb) is present (the word right before
"migrated" is "they", not "i"/"we").

Each qualifying sentence also gets a build/buy stance tag (config
sentiment._stance_rule) so the per-sentence tone can be grouped by decision
direction downstream (step4 writes decision_sentences.csv, one row per
sentence, for exactly this - e.g. a tone-by-stance box plot).

decision_tone_mean / n_decision_sentences remain in the output as a
convenience per-thread summary; the un-collapsed detail is in the
'sentences' list.

All dictionaries live in config/keywords.json like every other instrument
(dictionary-based content analysis: Krippendorff 2018; Grimmer & Stewart
2013). No other sentiment is measured anywhere in the pipeline.

Standalone use:
  python3 -m pipeline.step3d_sentiment --config config/keywords.json --out out
"""

import argparse
import os
import re

from . import common
from .common import (PHRASE_GAP_ADVERBS, build_sentiment, compile_terms,
                     count_matches, sentences_of)


class SentimentMeasurer:
    def __init__(self, config):
        self.score, self.backend = build_sentiment()
        senti = config["sentiment"]
        subjects = "|".join(re.escape(s) for s in senti["decision_subjects"])
        # longest-first: "built internally" must win over the bare "built"
        # prefix it contains, or the regex would stop at "built" (a word
        # boundary sits right before the following space either way).
        verbs = "|".join(re.escape(v) for v in
                         sorted(senti["decision_verbs"], key=len, reverse=True))
        adverbs = "|".join(re.escape(a) for a in PHRASE_GAP_ADVERBS)
        self.re_subject_verb = re.compile(
            rf"\b(?:{subjects})(?:\s+(?:{adverbs}))?\s+(?P<verb>{verbs})\b",
            re.IGNORECASE)

        provider_terms = []
        for entry in config["providers"]["entries"]:
            provider_terms.extend(entry.get("patterns", []))
            provider_terms.extend(entry.get("ambiguous_patterns", []))
        self.re_context = compile_terms(config["gate"]["legaltech_terms"] + provider_terms)

        stance = config["stance"]
        self.re_build_signal = compile_terms(stance["build_signals"])
        self.re_buy_signal = compile_terms(stance["buy_signals"])
        self.re_provider_named = compile_terms(provider_terms)
        self.build_verbs = {v.lower() for v in senti.get("unambiguous_build_verbs", [])}
        self.buy_verbs = {v.lower() for v in senti.get("unambiguous_buy_verbs", [])}

    def _stance_of(self, sentence, verb):
        """See config sentiment._stance_rule."""
        build_hit = bool(count_matches(self.re_build_signal, sentence))
        buy_hit = bool(count_matches(self.re_buy_signal, sentence))
        if not build_hit and not buy_hit:
            verb_l = verb.lower()
            if verb_l in self.build_verbs:
                build_hit = True
            elif verb_l in self.buy_verbs or count_matches(self.re_provider_named, sentence):
                buy_hit = True
        if build_hit and buy_hit:
            return "mixed"
        if build_hit:
            return "build"
        if buy_hit:
            return "buy"
        return "unclear"

    def decision_sentences(self, op_text, comment_texts):
        """[(sentence, verb, stance)] for sentences (OP + comments) that
        contain a direct subject-verb bigram AND a legaltech/provider
        context term in the same sentence - see module docstring."""
        full_text = " \n ".join([op_text] + comment_texts)
        hits = []
        for sentence in sentences_of(full_text):
            match = self.re_subject_verb.search(sentence)
            if not match or not count_matches(self.re_context, sentence):
                continue
            verb = match.group("verb")
            hits.append((sentence, verb, self._stance_of(sentence, verb)))
        return hits

    def measure(self, op_text, comment_texts):
        hits = self.decision_sentences(op_text, comment_texts)
        tones = [round(self.score(sentence), 4) for sentence, _verb, _stance in hits]
        mean = round(sum(tones) / len(tones), 4) if tones else None
        return {
            "decision_tone_mean": mean,
            "n_decision_sentences": len(tones),
            "sentences": [
                {"sentence": sentence, "verb": verb, "stance": stance, "tone": tone}
                for (sentence, verb, stance), tone in zip(hits, tones)
            ],
        }


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
