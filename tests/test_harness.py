"""Tests for reasoning_eval_harness. Run: python -m unittest discover -s tests"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reasoning_eval_harness import (
    EvalHarness,
    EvalReport,
    ModelResponse,
    RubricCriterion,
    TaskCase,
    classify_failure,
    keyword_overlap_grader,
    rubric_grader,
)


def make_cases():
    return [
        TaskCase(
            case_id="c1",
            prompt="p1",
            gold_answer="the quick brown fox",
            domain="math",
            tags=["arithmetic"],
            rubric=[RubricCriterion("names animal", ["fox"], 1.0)],
        ),
        TaskCase(
            case_id="c2",
            prompt="p2",
            gold_answer="gold two",
            domain="math",
            rubric=[RubricCriterion("says gold", ["gold"], 1.0)],
        ),
        TaskCase(
            case_id="c3",
            prompt="p3",
            gold_answer="gold three",
            domain="logic",
            rubric=[RubricCriterion("says gold", ["gold"], 1.0)],
        ),
        TaskCase(
            case_id="c4",
            prompt="p4",
            gold_answer="gold four",
            domain="logic",
            rubric=[RubricCriterion("says gold", ["gold"], 1.0)],
        ),
    ]


class TestModelResponse(unittest.TestCase):
    def test_confidence_bounds(self):
        with self.assertRaises(ValueError):
            ModelResponse("a", 1.5)
        with self.assertRaises(ValueError):
            ModelResponse("a", -0.1)
        with self.assertRaises(ValueError):
            ModelResponse("a", "high")
        r = ModelResponse("a", 0)
        self.assertEqual(r.confidence, 0.0)


class TestGraders(unittest.TestCase):
    def test_rubric_grader_full_and_partial(self):
        case = TaskCase(
            "x", "p", "gold",
            rubric=[
                RubricCriterion("a", ["alpha", "beta"], 0.5),
                RubricCriterion("b", ["gamma"], 0.5),
            ],
        )
        self.assertEqual(
            rubric_grader(case, ModelResponse("alpha beta gamma", 0.9)), 1.0
        )
        # one of two keywords in first criterion -> 0.5 * 0.5 = 0.25,
        # second criterion missed -> 0.0
        self.assertAlmostEqual(
            rubric_grader(case, ModelResponse("alpha only", 0.9)), 0.25
        )
        self.assertEqual(rubric_grader(case, ModelResponse("nothing", 0.9)), 0.0)

    def test_rubric_grader_falls_back_without_rubric(self):
        case = TaskCase("x", "p", "the quick brown fox jumps")
        score = rubric_grader(case, ModelResponse("brown fox jumps", 0.5))
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_keyword_overlap_empty_gold(self):
        case = TaskCase("x", "p", "a an the")
        self.assertEqual(
            keyword_overlap_grader(case, ModelResponse("anything", 0.5)), 0.0
        )


class TestClassify(unittest.TestCase):
    def test_quadrants(self):
        self.assertEqual(classify_failure(0.9, 0.9), "calibrated_success")
        self.assertEqual(classify_failure(0.9, 0.2), "underconfident_success")
        self.assertEqual(classify_failure(0.2, 0.9), "overconfident_error")
        self.assertEqual(classify_failure(0.2, 0.2), "calibrated_error")


class TestHarness(unittest.TestCase):
    def test_register_validation(self):
        h = EvalHarness()
        with self.assertRaises(ValueError):
            h.register_suite("", make_cases())
        with self.assertRaises(ValueError):
            h.register_suite("s", [])
        dup = make_cases() + [make_cases()[0]]
        with self.assertRaises(ValueError):
            h.register_suite("s", dup)
        with self.assertRaises(TypeError):
            h.register_suite("s", ["not-a-case"])

    def test_unknown_suite(self):
        h = EvalHarness()
        h.register_suite("s", make_cases())
        with self.assertRaises(KeyError):
            h.run(lambda p: ModelResponse("x", 0.5), "missing")

    def test_model_must_return_model_response(self):
        h = EvalHarness()
        h.register_suite("s", make_cases())
        with self.assertRaises(TypeError):
            h.run(lambda p: ("answer", 0.5), "s")

    def test_accuracy_and_ece(self):
        # scores: c1..c3 pass (1.0), c4 fails (0.0)
        # confs: 0.9, 0.9, 0.1, 0.1
        answers = {
            "p1": ("fox", 0.9),
            "p2": ("gold", 0.9),
            "p3": ("gold", 0.1),
            "p4": ("wrong", 0.1),
        }
        h = EvalHarness()
        h.register_suite("s", make_cases())
        report = h.run(
            lambda p: ModelResponse(*answers[p]), "s"
        )
        self.assertAlmostEqual(report.accuracy, 0.75)
        self.assertAlmostEqual(report.mean_score, 0.75)
        self.assertAlmostEqual(report.mean_confidence, 0.5)
        # bin [0,0.2): n=2, conf=0.1, acc=0.5 ; bin [0.8,1.0]: n=2, conf=0.9, acc=1.0
        # ECE = 0.5*|0.5-0.1| + 0.5*|1.0-0.9| = 0.2 + 0.05 = 0.25
        self.assertAlmostEqual(report.expected_calibration_error, 0.25)
        tax = report.failure_taxonomy()
        self.assertEqual(tax["calibrated_success"], 2)
        self.assertEqual(tax["underconfident_success"], 1)
        self.assertEqual(tax["calibrated_error"], 1)
        self.assertNotIn("overconfident_error", tax)
        dom = report.domain_accuracy()
        self.assertAlmostEqual(dom["math"][0], 1.0)
        self.assertAlmostEqual(dom["logic"][0], 0.5)
        # only failing case is c4, which has no tags
        self.assertEqual(report.tag_failure_counts(), {})

    def test_tag_failure_counts(self):
        cases = make_cases()
        h = EvalHarness()
        h.register_suite("s", cases)
        report = h.run(lambda p: ModelResponse("unrelated", 0.9), "s")
        self.assertEqual(report.accuracy, 0.0)
        self.assertEqual(report.tag_failure_counts(), {"arithmetic": 1})
        self.assertEqual(
            report.failure_taxonomy()["overconfident_error"], 4
        )

    def test_custom_grader(self):
        h = EvalHarness()
        h.register_suite("s", make_cases())
        report = h.run(
            lambda p: ModelResponse("x", 0.5), "s",
            grader=lambda case, resp: 1.0,
        )
        self.assertEqual(report.accuracy, 1.0)

    def test_run_all_and_summary(self):
        h = EvalHarness()
        h.register_suite("a", make_cases()[:2])
        h.register_suite("b", make_cases()[2:])
        reports = h.run_all(lambda p: ModelResponse("gold", 0.8))
        self.assertEqual(sorted(reports), ["a", "b"])
        text = reports["a"].summary()
        for needle in (
            "Eval report: a", "accuracy", "expected calibration error",
            "failure taxonomy", "accuracy by domain",
        ):
            self.assertIn(needle, text)

    def test_save_report_roundtrip(self):
        import json
        import tempfile

        h = EvalHarness()
        h.register_suite("s", make_cases())
        report = h.run(lambda p: ModelResponse("gold", 0.8), "s")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            path = fh.name
        h.save_report(report, path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["suite_name"], "s")
        self.assertEqual(data["n_cases"], 4)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
