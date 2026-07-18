#!/usr/bin/env python3
"""Regenerate tests/expected/fixture_analysis.expected.json — USE DELIBERATELY.

The self-test (pipeline/checks.py) exists to catch ACCIDENTAL changes to the
analysis code: it re-analyses the frozen fixture thread with every analysis
module and compares each result with the frozen expected model. Run this
script ONLY after an INTENTIONAL method change (e.g. you changed the tier
rule or the stance thresholds on purpose), to freeze the new behaviour as the
expected model. Document the change and the re-freeze date in the thesis
methodology.

Usage:  python3 tests/freeze_expected_fixture.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import common                                    # noqa: E402
from pipeline.step2_tier import TierAssigner                   # noqa: E402
from pipeline.step3a_stance import StanceMeasurer              # noqa: E402
from pipeline.step3b_factors import FactorMeasurer             # noqa: E402
from pipeline.step3c_providers import ProviderMeasurer         # noqa: E402
from pipeline.step3d_sentiment import SentimentMeasurer        # noqa: E402
from pipeline.step3e_terms import TermCounter                  # noqa: E402


def main():
    fixture = common.read_json(os.path.join(ROOT, "tests", "fixtures", "fixture_thread.json"))
    config = common.read_json(os.path.join(ROOT, "tests", "fixtures", "fixture_config.json"))

    title, selftext, _comments, _meta = common.load_wrapper_texts(fixture)
    tier, reason = TierAssigner(config).tier(title, selftext)

    op_text, comments = common.extract_op_and_comments(fixture)
    sentiment_measurer = SentimentMeasurer(config)
    term_counter = TermCounter(config)
    counts, _surfaces = term_counter.count(" \n ".join([op_text] + comments))

    expected = {
        "_readme": ("Frozen 'ideal output' for the pipeline self-test: what the tier rule and "
                    "each analysis module must produce for tests/fixtures/fixture_thread.json "
                    "under tests/fixtures/fixture_config.json. pipeline/checks.py compares "
                    "against this EXACTLY before every run. Regenerate only deliberately, after "
                    "an intentional method change, with tests/freeze_expected_fixture.py."),
        "sentiment_backend": sentiment_measurer.backend,
        "stemmer": term_counter.stem_backend,
        "tier": tier,
        "tier_reason": reason,
        "stance": StanceMeasurer(config).measure(op_text, comments),
        "factors": FactorMeasurer(config).measure(op_text, comments),
        "providers": ProviderMeasurer(config).measure(op_text, comments),
        "sentiment": sentiment_measurer.measure(op_text, comments),
        "term_counts": dict(counts),
    }
    out_path = os.path.join(ROOT, "tests", "expected", "fixture_analysis.expected.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(expected, fh, indent=2, ensure_ascii=False)
    print(f"💾 froze new expected model: {out_path}")
    print(f"   tier {tier} ({reason}); stance {expected['stance']['stance']}; "
          f"{len(expected['factors'])} factors; {len(expected['providers'])} providers; "
          f"{expected['sentiment']['n_decision_sentences']} decision sentence(s); "
          f"{len(expected['term_counts'])} stems")
    print("   ⚠️ Only do this after an INTENTIONAL method change — document it in the thesis.")


if __name__ == "__main__":
    main()
