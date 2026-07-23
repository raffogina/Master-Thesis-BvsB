"""Automatic quality gates run between pipeline steps.

Two kinds of test, both compared against files in tests/ (the "ideal expected
output models"):

1. SELF-TEST (before anything runs): a frozen fixture thread
   (tests/fixtures/fixture_thread.json) with known content is pushed through
   the tier rule and every analysis module using a frozen config snapshot
   (tests/fixtures/fixture_config.json). Each module's result must match
   tests/expected/fixture_analysis.expected.json EXACTLY — if analysis code
   was accidentally changed, this catches it before any data is touched.

2. OUTPUT CHECKS (after each step): the step's real output file is validated
   against its expected-output model in tests/expected/*.expected.json
   (exact column set/order, row minimums, unique keys, allowed values, numeric
   ranges) plus cross-step consistency rules coded below (e.g. every analysis
   module must cover exactly the threads step 2 kept; percentages 0-100;
   sentiment within VADER's [-1, 1]).

Failure semantics (run_pipeline.py):
  - backbone failure (collect/tier/aggregate, or the tier self-test) stops
    the whole pipeline — nothing downstream is possible;
  - an ANALYSIS module that fails its self-test, crashes, or fails its output
    gate is QUARANTINED: its results are excluded, every other module still
    runs and the final tables are still produced (without that dimension);
  - an analysis module whose file was deliberately DELETED is treated as
    "absent by choice": noted, skipped, not an error.

Every result is written to out/checks_report.txt for the thesis audit trail.

Standalone use (validate whatever output files currently exist):
  python3 -m pipeline.checks --out out --config config/keywords.json
"""

import argparse
import csv
import importlib
import os

from . import common
from .step2_tier import TierAssigner

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPECTED_DIR = os.path.join(REPO_ROOT, "tests", "expected")
DEFAULT_FIXTURES_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")

# module key -> (python module name, output file, expected-model file)
ANALYSIS_MODULES = {
    "stance": ("step3a_stance", common.STANCE_NAME, "step3a_stance.expected.json"),
    "factors": ("step3b_factors", common.FACTORS_NAME, "step3b_factors.expected.json"),
    "providers": ("step3c_providers", common.PROVIDERS_NAME, "step3c_providers.expected.json"),
    "sentiment": ("step3d_sentiment", common.SENTIMENT_NAME, "step3d_sentiment.expected.json"),
    "terms": ("step3e_terms", common.TERM_COUNTS_NAME, "step3e_term_counts.expected.json"),
}


def import_analysis_module(module):
    """(imported module, status) where status is 'ok', 'absent' or an error
    message. 'absent' = the .py file was deleted on purpose (supported)."""
    pyname = ANALYSIS_MODULES[module][0]
    try:
        return importlib.import_module(f"pipeline.{pyname}"), "ok"
    except ModuleNotFoundError as exc:
        if exc.name == f"pipeline.{pyname}":
            return None, "absent"
        return None, f"import failed: {exc}"
    except Exception as exc:   # SyntaxError etc. = a bug in that module only
        return None, f"import failed: {exc}"


def _read_csv_with_header(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return reader.fieldnames or [], rows


class Checker:
    """Collects check results per stage and writes the report file."""

    def __init__(self, out_dir, config_path,
                 expected_dir=DEFAULT_EXPECTED_DIR,
                 fixtures_dir=DEFAULT_FIXTURES_DIR):
        self.out_dir = out_dir
        self.config = common.load_config(config_path)
        self.expected_dir = expected_dir
        self.fixtures_dir = fixtures_dir
        self.results = []          # (stage, check_name, passed, detail)

    # ------------------------------------------------------------------ utils
    def _record(self, stage, name, passed, detail=""):
        self.results.append((stage, name, passed, detail))
        mark = "✅" if passed else "❌"
        print(f"   {mark} {name}" + (f" — {detail}" if detail and not passed else ""))
        return passed

    def _spec(self, spec_name):
        return common.read_json(os.path.join(self.expected_dir, spec_name))

    def stage_passed(self, stage):
        return all(ok for s, _n, ok, _d in self.results if s == stage)

    def total_counts(self):
        total = len(self.results)
        failed = sum(1 for _s, _n, ok, _d in self.results if not ok)
        return total, total - failed, failed

    # ------------------------------------------- generic spec-driven checking
    def _check_csv_against_spec(self, stage, path, spec):
        """Validate one CSV output against its expected-output model."""
        label = os.path.basename(path)
        if not os.path.exists(path):
            self._record(stage, f"{label}: file exists", False, f"missing: {path}")
            return None
        header, rows = _read_csv_with_header(path)

        self._record(stage, f"{label}: columns match expected model",
                     header == spec["columns"],
                     f"got {header}, expected {spec['columns']}")
        self._record(stage, f"{label}: at least {spec.get('min_rows', 0)} row(s)",
                     len(rows) >= spec.get("min_rows", 0),
                     f"got {len(rows)} rows")

        for col in spec.get("unique_columns", []):
            values = [r.get(col, "") for r in rows]
            dupes = len(values) - len(set(values))
            self._record(stage, f"{label}: '{col}' values are unique",
                         dupes == 0, f"{dupes} duplicated value(s)")

        for col in spec.get("non_empty_columns", []):
            n_empty = sum(1 for r in rows if not (r.get(col) or "").strip())
            self._record(stage, f"{label}: '{col}' never empty",
                         n_empty == 0, f"{n_empty} empty value(s)")

        for col, allowed in spec.get("allowed_values", {}).items():
            bad = sorted({r.get(col, "") for r in rows} - set(allowed), key=str)
            self._record(stage, f"{label}: '{col}' only contains expected values",
                         not bad, f"unexpected value(s): {bad}")

        for col, rng in spec.get("numeric_ranges", {}).items():
            bad = 0
            for r in rows:
                raw = (r.get(col) or "").strip()
                if raw == "":
                    if not rng.get("allow_empty", False):
                        bad += 1
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    bad += 1
                    continue
                if val < rng.get("min", float("-inf")) or val > rng.get("max", float("inf")):
                    bad += 1
            self._record(stage, f"{label}: '{col}' within expected numeric range",
                         bad == 0, f"{bad} out-of-range/invalid value(s)")
        return rows

    def _check_json_against_spec(self, stage, path, spec):
        """Validate one JSON output against its expected-output model."""
        label = os.path.basename(path)
        if not os.path.exists(path):
            self._record(stage, f"{label}: file exists", False, f"missing: {path}")
            return None
        payload = common.read_json(path)
        records = payload.get(spec["records_key"], [])

        self._record(stage, f"{label}: at least {spec.get('min_records', 0)} record(s)",
                     len(records) >= spec.get("min_records", 0),
                     f"got {len(records)}")

        missing = 0
        for rec in records:
            for field in spec.get("required_fields", []):
                if field not in rec:
                    missing += 1
        self._record(stage, f"{label}: every record has the expected fields",
                     missing == 0, f"{missing} missing field occurrence(s)")

        for field, allowed in spec.get("allowed_values", {}).items():
            bad = sorted({str(rec.get(field)) for rec in records} - set(allowed), key=str)
            self._record(stage, f"{label}: '{field}' only contains expected values",
                         not bad, f"unexpected value(s): {bad}")

        for field, rng in spec.get("numeric_ranges", {}).items():
            bad = 0
            for rec in records:
                val = rec.get(field)
                if val is None:
                    if not rng.get("nullable", False):
                        bad += 1
                    continue
                if not isinstance(val, (int, float)):
                    bad += 1
                    continue
                if val < rng.get("min", float("-inf")) or val > rng.get("max", float("inf")):
                    bad += 1
            self._record(stage, f"{label}: '{field}' within expected numeric range",
                         bad == 0, f"{bad} out-of-range/invalid value(s)")
        return payload

    # ------------------------------------------------------------- self-test
    def self_test(self):
        """Analyse the frozen fixture thread with every analysis module and compare
        with the frozen expected model. Uses the config SNAPSHOT shipped with
        the fixture, so later edits to config/keywords.json (e.g. adding
        factor terms after a step-B review) do not break the test: it
        validates the CODE, not the current dictionaries.

        Returns {"tier"|module: "ok"|"failed"|"absent"}.
        """
        stage = "SELF-TEST"
        print("🧪 Self-test — fixture thread vs frozen expected model")
        statuses = {}
        try:
            fixture = common.read_json(os.path.join(self.fixtures_dir, "fixture_thread.json"))
            fixture_config = common.read_json(os.path.join(self.fixtures_dir, "fixture_config.json"))
            expected = self._spec("fixture_analysis.expected.json")
        except (OSError, ValueError) as exc:
            self._record(stage, "fixture and expected-model files readable",
                         False, f"{exc} — restore the tests/ folder from git")
            return {"tier": "failed"}

        texts = common.extract_op_and_comments(fixture)
        if texts is None:
            self._record(stage, "fixture readable", False,
                         "extract_op_and_comments returned None")
            return {"tier": "failed"}
        op_text, comments = texts

        # -- tier rule (backbone, step 2 — never optional) ---------------------
        title, selftext, _c, _m = common.load_wrapper_texts(fixture)
        tier, reason = TierAssigner(fixture_config).tier(title, selftext)
        ok = self._record(stage, "corpus tier matches expected model",
                          tier == expected["tier"],
                          f"got {tier}, expected {expected['tier']}")
        ok = self._record(stage, "tier reason matches expected model",
                          reason == expected["tier_reason"],
                          f"got '{reason}', expected '{expected['tier_reason']}'") and ok
        statuses["tier"] = "ok" if ok else "failed"

        # -- analysis modules (each optional, each tested on its own) ----------
        def module_status(module, test_fn):
            mod, status = import_analysis_module(module)
            if status == "absent":
                self._record(stage, f"{module} module absent — will be skipped "
                                    "(supported: its columns/tables stay empty)", True)
                return "absent"
            if mod is None:
                self._record(stage, f"{module} module imports", False, status)
                return "failed"
            try:
                return "ok" if test_fn(mod) else "failed"
            except Exception as exc:
                self._record(stage, f"{module} self-test runs without crashing",
                             False, repr(exc))
                return "failed"

        def test_stance(mod):
            got = mod.StanceMeasurer(fixture_config).measure(op_text, comments)
            return self._record(stage, "stance matches expected model",
                                got == expected["stance"],
                                f"got {got!r}, expected {expected['stance']!r}")

        def test_factors(mod):
            got = mod.FactorMeasurer(fixture_config).measure(op_text, comments)
            return self._record(stage, "factor mentions match expected model",
                                got == expected["factors"],
                                f"got {got!r}, expected {expected['factors']!r}")

        def test_providers(mod):
            got = mod.ProviderMeasurer(fixture_config).measure(op_text, comments)
            return self._record(stage, "provider mentions match expected model",
                                got == expected["providers"],
                                f"got {got!r}, expected {expected['providers']!r}")

        def test_sentiment(mod):
            got = mod.SentimentMeasurer(fixture_config).measure(op_text, comments)
            return self._record(stage, "decision-sentence tone matches expected model",
                                got == expected["sentiment"],
                                f"got {got!r}, expected {expected['sentiment']!r}")

        def test_terms(mod):
            counter = mod.TermCounter(fixture_config)
            if counter.stem_backend != expected["stemmer"]:
                return self._record(stage, "stemmer differs from expected model "
                                           "(term-count comparison skipped)", True,
                                    f"{counter.stem_backend} vs {expected['stemmer']}")
            full_text = " \n ".join([op_text] + comments)
            counts, _surfaces = counter.count(full_text)
            return self._record(stage, "stemmed term counts match expected model",
                                dict(counts) == expected["term_counts"],
                                "stem counters differ from frozen model")

        tests = {"stance": test_stance, "factors": test_factors,
                 "providers": test_providers, "sentiment": test_sentiment,
                 "terms": test_terms}
        for module, test_fn in tests.items():
            statuses[module] = module_status(module, test_fn)
        return statuses

    # ------------------------------------------------------------ step checks
    def check_step1(self):
        stage = "STEP 1"
        print("🔎 Checking step 1 output against expected model")
        sdir = common.steps_dir(self.out_dir)
        rows = self._check_csv_against_spec(
            stage, os.path.join(sdir, common.MANIFEST_NAME),
            self._spec("step1_manifest.expected.json"))
        if rows:
            missing = sum(1 for r in rows if not os.path.exists(r["cache_file"]))
            self._record(stage, "manifest: every cache_file exists on disk",
                         missing == 0, f"{missing} missing file(s)")
        params_path = os.path.join(sdir, common.RUN_PARAMS_NAME)
        ok = os.path.exists(params_path)
        self._record(stage, f"{common.RUN_PARAMS_NAME}: file exists", ok, params_path)
        if ok:
            params = common.read_json(params_path)
            self._record(stage, "run params record queries and subreddits",
                         bool(params.get("queries")) and bool(params.get("subreddits")),
                         "queries/subreddits missing or empty")
        return self.stage_passed(stage)

    def check_step2(self):
        stage = "STEP 2"
        print("🔎 Checking step 2 output against expected model")
        sdir = common.steps_dir(self.out_dir)
        rows = self._check_csv_against_spec(
            stage, os.path.join(sdir, common.TIERS_NAME),
            self._spec("step2_tiers.expected.json"))
        if rows is not None:
            manifest_ids = {r["thread_id"]
                            for r in common.read_csv(os.path.join(sdir, common.MANIFEST_NAME))}
            tier_ids = {r["thread_id"] for r in rows}
            self._record(stage, "tiers cover exactly the manifest threads "
                                "(none lost, none invented)",
                         manifest_ids == tier_ids,
                         f"missing {sorted(manifest_ids - tier_ids)[:5]}, "
                         f"extra {sorted(tier_ids - manifest_ids)[:5]}")
            kept = sum(1 for r in rows if r["corpus_tier"] in ("1", "2"))
            self._record(stage, "at least one thread kept (tier 1 or 2)",
                         kept >= 1, "0 threads survived the gate — check the "
                                    "gate dictionaries / collection")
        return self.stage_passed(stage)

    def _kept_ids(self, tiers=("1", "2")):
        tiers_path = os.path.join(common.steps_dir(self.out_dir), common.TIERS_NAME)
        return [r["thread_id"] for r in common.read_csv(tiers_path)
                if r["corpus_tier"] in tiers]

    def check_module(self, module):
        """Output gate for one analysis module (stance/factors/providers/
        sentiment/terms)."""
        pyname, out_name, spec_name = ANALYSIS_MODULES[module]
        stage = f"STEP 3 ({module})"
        print(f"🔎 Checking {module} output against expected model")
        path = os.path.join(common.steps_dir(self.out_dir), out_name)
        payload = self._check_json_against_spec(stage, path, self._spec(spec_name))
        if payload is None:
            return False
        records = payload.get("threads", [])

        got_ids = [r.get("thread_id", "") for r in records]
        if module == "sentiment":
            # the sentiment module measures decision tone on tier-1 threads only
            kept_ids = self._kept_ids(("1",))
            self._record(stage, "sentiment: covers exactly the tier-1 threads "
                                "step 2 kept",
                         got_ids == kept_ids,
                         f"{len(got_ids)} measured vs {len(kept_ids)} tier-1 kept")
        else:
            kept_ids = self._kept_ids()
            self._record(stage, f"{module}: covers exactly the threads step 2 kept",
                         got_ids == kept_ids,
                         f"{len(got_ids)} measured vs {len(kept_ids)} kept")

        factor_keys = set(common.compile_factor_patterns(self.config).keys())
        provider_names = {p[0] for p in common.compile_provider_patterns(self.config)}

        if module == "factors":
            bad_key, bad_entry = [], 0
            for rec in records:
                for key, entry in rec.get("factors", {}).items():
                    if key not in factor_keys:
                        bad_key.append(key)
                    if not (isinstance(entry, list) and len(entry) == 2
                            and isinstance(entry[1], int) and entry[1] >= 1):
                        bad_entry += 1
            self._record(stage, "all factors exist in the config dictionary",
                         not bad_key, f"unknown factor(s): {sorted(set(bad_key))}")
            self._record(stage, "factor entries well-formed ([label, mentions >= 1])",
                         bad_entry == 0, f"{bad_entry} malformed entr(y/ies)")

        if module == "providers":
            bad_name, bad_entry = [], 0
            for rec in records:
                for name, entry in rec.get("providers", {}).items():
                    if name not in provider_names:
                        bad_name.append(name)
                    if not (isinstance(entry, list) and len(entry) == 2
                            and isinstance(entry[1], int) and entry[1] >= 1):
                        bad_entry += 1
            self._record(stage, "all providers exist in the config dictionary",
                         not bad_name, f"unknown provider(s): {sorted(set(bad_name))}")
            self._record(stage, "provider entries well-formed ([category, mentions >= 1])",
                         bad_entry == 0, f"{bad_entry} malformed entr(y/ies)")

        if module == "sentiment":
            valid_stances = {"build", "buy", "mixed", "unclear"}
            bad = 0
            for rec in records:
                tone = rec.get("decision_tone_mean")
                n = rec.get("n_decision_sentences")
                sentences = rec.get("sentences", [])
                well_formed = (
                    isinstance(n, int) and n >= 0 and len(sentences) == n
                    and ((tone is None and n == 0)
                        or (isinstance(tone, (int, float)) and -1 <= tone <= 1 and n >= 1))
                    and all(isinstance(e.get("tone"), (int, float))
                           and -1 <= e["tone"] <= 1
                           and e.get("stance") in valid_stances
                           for e in sentences))
                if not well_formed:
                    bad += 1
            self._record(stage, "decision tones within [-1, 1], consistent with "
                                "the sentence counts, and sentence list well-formed",
                         bad == 0, f"{bad} malformed record(s)")

        if module == "terms":
            bad_counts = 0
            for rec in records:
                for stem, count in rec.get("counts", {}).items():
                    surfaces = rec.get("surfaces", {}).get(stem, {})
                    if (not isinstance(count, int) or count < 1
                            or sum(surfaces.values()) != count):
                        bad_counts += 1
            self._record(stage, "term counts internally consistent "
                                "(surface-form counts sum to each stem count)",
                         bad_counts == 0, f"{bad_counts} inconsistent stem(s)")
        return self.stage_passed(stage)

    def check_step4(self, used_modules):
        """used_modules: analysis modules whose outputs step 4 consumed —
        only their tables are validated."""
        stage = "STEP 4"
        print("🔎 Checking step 4 outputs against expected models")
        min_count = self.config["terms"]["min_count"]

        master = self._check_csv_against_spec(
            stage, os.path.join(self.out_dir, "threads_master.csv"),
            self._spec("step4_threads_master.expected.json"))
        if master is not None:
            master_ids = [r.get("thread_id", "") for r in master]
            kept_ids = self._kept_ids()
            self._record(stage, "master table lists exactly the kept threads, "
                                "in step-2 order",
                         master_ids == kept_ids,
                         f"{len(master_ids)} vs {len(kept_ids)} rows")

        if "factors" in used_modules:
            factors = self._check_csv_against_spec(
                stage, os.path.join(self.out_dir, "factor_salience.csv"),
                self._spec("step4_factor_salience.expected.json"))
            if factors is not None:
                factor_keys = set(common.compile_factor_patterns(self.config).keys())
                bad = sorted({r.get("factor", "") for r in factors} - factor_keys)
                self._record(stage, "factor_salience: factors exist in the config",
                             not bad, f"unknown: {bad}")
                n_bad = sum(1 for r in factors
                            if int(r.get("n_threads_decision") or 0)
                            + int(r.get("n_threads_discourse") or 0)
                            != int(r.get("n_threads_all") or -1))
                self._record(stage, "factor_salience: decision + discourse counts "
                                    "add up to the total", n_bad == 0, f"{n_bad} row(s) off")

        if "providers" in used_modules:
            providers = self._check_csv_against_spec(
                stage, os.path.join(self.out_dir, "provider_mentions.csv"),
                self._spec("step4_provider_mentions.expected.json"))
            if providers is not None:
                provider_names = {p[0] for p in common.compile_provider_patterns(self.config)}
                bad = sorted({r.get("provider", "") for r in providers} - provider_names)
                self._record(stage, "provider_mentions: providers exist in the config",
                             not bad, f"unknown: {bad}")

        if "terms" in used_modules:
            discovery = self._check_csv_against_spec(
                stage, os.path.join(self.out_dir, "term_discovery.csv"),
                self._spec("step4_term_discovery.expected.json"))
            if discovery is not None:
                bad = sum(1 for r in discovery if int(r.get("total_count") or 0) < min_count)
                self._record(stage, f"term_discovery: totals >= config min_count "
                                    f"({min_count})", bad == 0, f"{bad} row(s) below")

        if "sentiment" in used_modules:
            self._check_csv_against_spec(
                stage, os.path.join(self.out_dir, "decision_sentences.csv"),
                self._spec("step4_decision_sentences.expected.json"))

        summary_path = os.path.join(self.out_dir, "run_summary.txt")
        self._record(stage, "run_summary.txt exists and is non-empty",
                     os.path.exists(summary_path) and os.path.getsize(summary_path) > 0,
                     summary_path)
        return self.stage_passed(stage)

    # -------------------------------------------------------------- reporting
    def write_report(self):
        path = os.path.join(self.out_dir, "checks_report.txt")
        total, passed, failed = self.total_counts()
        os.makedirs(self.out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"Pipeline checks report — {common.utc_now_iso()}\n")
            fh.write(f"Expected-output models: {self.expected_dir}\n\n")
            current = None
            for stage, name, ok, detail in self.results:
                if stage != current:
                    fh.write(f"[{stage}]\n")
                    current = stage
                fh.write(f"  {'PASS' if ok else 'FAIL'}  {name}\n")
                if not ok and detail:
                    fh.write(f"        -> {detail}\n")
            fh.write(f"\nTotal: {total} checks — {passed} passed, {failed} failed\n")
        print(f"   💾 {path} ({passed}/{total} checks passed)")
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/keywords.json")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()

    checker = Checker(args.out, args.config)
    checker.self_test()
    checker.check_step1()
    checker.check_step2()
    present = []
    for module, (_py, out_name, _spec) in ANALYSIS_MODULES.items():
        if os.path.exists(os.path.join(common.steps_dir(args.out), out_name)):
            checker.check_module(module)
            present.append(module)
        else:
            print(f"   ℹ️ {module} output not present — skipping its checks")
    checker.check_step4(set(present))
    ok = checker.write_report()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
