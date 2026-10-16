"""reasoning_eval_harness.harness

Register task suites, run any Python model callable, and get accuracy,
calibration (confidence vs. accuracy), and a failure-taxonomy report.

A "model" is any callable with the signature::

    model(prompt: str) -> ModelResponse

Standard library only. No network access, no third-party dependencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

PASS_THRESHOLD = 0.7
HIGH_CONFIDENCE = 0.6
N_CALIBRATION_BINS = 5

_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been being "
    "it its this that these those as at by from into over after before "
    "not no nor but if then than so such can could should would may might "
    "will shall do does did have has had having he she they we you i my me "
    "him her them us my your his their our what which who whom whose when "
    "where why how all any both each few more most other some only own same "
    "too very".split()
)


@dataclass
class ModelResponse:
    """One model output. Confidence must be in [0, 1]."""

    answer: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, (int, float)):
            raise ValueError(
                "confidence must be numeric, got "
                + type(self.confidence).__name__
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                "confidence must be in [0, 1], got %r" % (self.confidence,)
            )
        self.confidence = float(self.confidence)


@dataclass
class RubricCriterion:
    """One graded criterion: weight * (fraction of keywords present)."""

    name: str
    keywords: List[str]
    weight: float


@dataclass
class TaskCase:
    """One evaluation case."""

    case_id: str
    prompt: str
    gold_answer: str
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    rubric: List[RubricCriterion] = field(default_factory=list)


Grader = Callable[[TaskCase, ModelResponse], float]


def rubric_grader(case: TaskCase, response: ModelResponse) -> float:
    """Score = sum over criteria of weight * keyword hit rate.

    A criterion scores the fraction of its keywords found (case-insensitive)
    in the answer. This is a heuristic proxy for rubric coverage, not a
    judgment of reasoning quality; treat scores as lower bounds on coverage.
    """
    if not case.rubric:
        return keyword_overlap_grader(case, response)
    answer = response.answer.lower()
    total = 0.0
    for criterion in case.rubric:
        if not criterion.keywords:
            continue
        hits = sum(1 for kw in criterion.keywords if kw.lower() in answer)
        total += criterion.weight * (hits / len(criterion.keywords))
    return max(0.0, min(1.0, total))


def keyword_overlap_grader(case: TaskCase, response: ModelResponse) -> float:
    """Fallback heuristic: fraction of distinctive gold terms in the answer."""
    gold_terms = {
        term
        for term in re.findall(r"[a-z]{4,}", case.gold_answer.lower())
        if term not in _STOPWORDS
    }
    if not gold_terms:
        return 0.0
    answer = response.answer.lower()
    hits = sum(1 for term in gold_terms if term in answer)
    return hits / len(gold_terms)


def classify_failure(score: float, confidence: float,
                     threshold: float = PASS_THRESHOLD) -> str:
    """Quadrant of the confidence/accuracy plane for one case."""
    passed = score >= threshold
    confident = confidence >= HIGH_CONFIDENCE
    if passed and confident:
        return "calibrated_success"
    if passed and not confident:
        return "underconfident_success"
    if not passed and confident:
        return "overconfident_error"
    return "calibrated_error"


@dataclass
class CaseResult:
    case: TaskCase
    response: ModelResponse
    score: float
    passed: bool
    failure_class: str

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case.case_id,
            "domain": self.case.domain,
            "tags": list(self.case.tags),
            "score": round(self.score, 4),
            "confidence": round(self.response.confidence, 4),
            "passed": self.passed,
            "failure_class": self.failure_class,
            "answer": self.response.answer,
        }


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    mean_accuracy: float


@dataclass
class EvalReport:
    suite_name: str
    results: List[CaseResult]
    threshold: float = PASS_THRESHOLD

    @property
    def n_cases(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        """Fraction of cases with score >= threshold."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def mean_confidence(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.response.confidence for r in self.results) / len(
            self.results
        )

    def calibration_bins(self,
                         n_bins: int = N_CALIBRATION_BINS) -> List[CalibrationBin]:
        """Bin confidences; compare mean confidence to mean accuracy per bin.

        With few cases per bin the per-bin numbers are noisy; the ECE is a
        rough summary, not a precise measurement. Empty bins are skipped.
        """
        bins: List[CalibrationBin] = []
        width = 1.0 / n_bins
        for i in range(n_bins):
            lo = i * width
            hi = (i + 1) * width
            in_bin = [
                r for r in self.results
                if (lo <= r.response.confidence < hi)
                or (i == n_bins - 1 and r.response.confidence == 1.0)
            ]
            if not in_bin:
                continue
            mean_conf = sum(r.response.confidence for r in in_bin) / len(in_bin)
            mean_acc = sum(1 for r in in_bin if r.passed) / len(in_bin)
            bins.append(CalibrationBin(lo, hi, len(in_bin), mean_conf, mean_acc))
        return bins

    @property
    def expected_calibration_error(self) -> float:
        """Sum over bins of (n_bin / N) * |acc_bin - conf_bin|."""
        if not self.results:
            return 0.0
        total = 0.0
        for b in self.calibration_bins():
            total += (b.n / len(self.results)) * abs(
                b.mean_accuracy - b.mean_confidence
            )
        return total

    def failure_taxonomy(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.results:
            counts[r.failure_class] = counts.get(r.failure_class, 0) + 1
        return counts

    def domain_accuracy(self) -> Dict[str, Tuple[float, int]]:
        by_domain: Dict[str, List[CaseResult]] = {}
        for r in self.results:
            by_domain.setdefault(r.case.domain, []).append(r)
        return {
            domain: (
                sum(1 for r in rs if r.passed) / len(rs),
                len(rs),
            )
            for domain, rs in by_domain.items()
        }

    def tag_failure_counts(self) -> Dict[str, int]:
        """Count failures per tag, to surface weak spots by concept."""
        counts: Dict[str, int] = {}
        for r in self.results:
            if r.passed:
                continue
            for tag in r.case.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def to_dict(self) -> Dict:
        return {
            "suite_name": self.suite_name,
            "n_cases": self.n_cases,
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, 4),
            "mean_score": round(self.mean_score, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "expected_calibration_error": round(
                self.expected_calibration_error, 4
            ),
            "calibration_bins": [
                {
                    "lo": round(b.lo, 2),
                    "hi": round(b.hi, 2),
                    "n": b.n,
                    "mean_confidence": round(b.mean_confidence, 4),
                    "mean_accuracy": round(b.mean_accuracy, 4),
                }
                for b in self.calibration_bins()
            ],
            "failure_taxonomy": self.failure_taxonomy(),
            "domain_accuracy": {
                d: {"accuracy": round(a, 4), "n": n}
                for d, (a, n) in self.domain_accuracy().items()
            },
            "tag_failure_counts": self.tag_failure_counts(),
            "cases": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [
            "Eval report: %s (n=%d)" % (self.suite_name, self.n_cases),
            "  accuracy (score >= %.2f): %.3f" % (self.threshold, self.accuracy),
            "  mean score: %.3f" % self.mean_score,
            "  mean confidence: %.3f" % self.mean_confidence,
            "  expected calibration error: %.3f" % self.expected_calibration_error,
            "  calibration bins (range: n, mean_conf -> mean_acc):",
        ]
        for b in self.calibration_bins():
            lines.append(
                "    %.1f-%.1f: n=%d, %.3f -> %.3f"
                % (b.lo, b.hi, b.n, b.mean_confidence, b.mean_accuracy)
            )
        lines.append("  failure taxonomy:")
        for name, count in sorted(self.failure_taxonomy().items()):
            lines.append("    %s: %d" % (name, count))
        lines.append("  accuracy by domain:")
        for domain, (acc, n) in sorted(self.domain_accuracy().items()):
            lines.append("    %s: %.3f (n=%d)" % (domain, acc, n))
        top_tags = list(self.tag_failure_counts().items())[:5]
        if top_tags:
            lines.append("  most frequent failure tags:")
            for tag, count in top_tags:
                lines.append("    %s: %d" % (tag, count))
        return "\n".join(lines)


class EvalHarness:
    """Registry of task suites plus the run loop."""

    def __init__(self, pass_threshold: float = PASS_THRESHOLD) -> None:
        self.pass_threshold = pass_threshold
        self._suites: Dict[str, List[TaskCase]] = {}

    def register_suite(self, name: str, cases: List[TaskCase]) -> None:
        if not name:
            raise ValueError("suite name must be non-empty")
        if not cases:
            raise ValueError("suite %r must contain at least one case" % name)
        seen = set()
        for case in cases:
            if not isinstance(case, TaskCase):
                raise TypeError("suite cases must be TaskCase instances")
            if case.case_id in seen:
                raise ValueError(
                    "duplicate case_id %r in suite %r" % (case.case_id, name)
                )
            seen.add(case.case_id)
        self._suites[name] = list(cases)

    @property
    def suite_names(self) -> List[str]:
        return sorted(self._suites)

    def run(self, model: Callable[[str], ModelResponse], suite_name: str,
            grader: Optional[Grader] = None) -> EvalReport:
        if suite_name not in self._suites:
            raise KeyError("unknown suite: %r" % suite_name)
        grade = grader or rubric_grader
        results: List[CaseResult] = []
        for case in self._suites[suite_name]:
            response = model(case.prompt)
            if not isinstance(response, ModelResponse):
                raise TypeError(
                    "model must return ModelResponse, got %s"
                    % type(response).__name__
                )
            score = grade(case, response)
            score = max(0.0, min(1.0, float(score)))
            passed = score >= self.pass_threshold
            results.append(
                CaseResult(
                    case=case,
                    response=response,
                    score=score,
                    passed=passed,
                    failure_class=classify_failure(
                        score, response.confidence, self.pass_threshold
                    ),
                )
            )
        return EvalReport(
            suite_name=suite_name, results=results,
            threshold=self.pass_threshold
        )

    def run_all(self, model: Callable[[str], ModelResponse],
                grader: Optional[Grader] = None) -> Dict[str, EvalReport]:
        return {
            name: self.run(model, name, grader) for name in self.suite_names
        }

    def save_report(self, report: EvalReport, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
            fh.write("\n")
