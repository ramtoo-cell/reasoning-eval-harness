# reasoning-eval-harness

A small, dependency-free harness for evaluating how well a language model
reasons: register task suites, run any Python model callable, and get
accuracy, calibration (confidence vs. accuracy), and a failure-taxonomy
report.

## Problem

Benchmarks usually report a single accuracy number. That hides two things
that matter for reasoning research:

1. **Calibration** — is the model confident exactly when it is right?
   An overconfident wrong answer is more dangerous than an uncertain one.
2. **Failure structure** — *where* does reasoning break down? A flat error
   rate does not tell you whether failures cluster on, say, incentive
   analysis versus legal classification.

## Method

A model is any Python callable `model(prompt: str) -> ModelResponse`, where
`ModelResponse` carries the answer and a self-reported confidence in [0, 1].

- **Suites.** Group `TaskCase`s (prompt, gold answer, domain, tags, optional
  rubric) into named suites with `EvalHarness.register_suite`.
- **Grading.** The default `rubric_grader` scores each rubric criterion by
  keyword hit-rate times its weight; without a rubric it falls back to a
  keyword-overlap heuristic. Bring your own `Grader` for anything smarter
  (e.g., an LLM judge) — the harness only requires a float in [0, 1].
- **Calibration.** Confidences are binned (5 bins); per bin the report shows
  mean confidence vs. mean accuracy, plus the expected calibration error
  (ECE = Σ (n_bin/N)·|acc − conf|).
- **Failure taxonomy.** Each case lands in one quadrant:
  `calibrated_success`, `underconfident_success`, `overconfident_error`,
  `calibrated_error`. Failures are also aggregated by tag so weak concepts
  surface (e.g., "arithmetic: 3 failures").

## Results

Measured by running `examples/demo.py` (a 6-case demo suite against a
deliberately imperfect stub model):

| Metric | Value |
|---|---|
| Accuracy (score ≥ 0.70) | 0.667 |
| Mean score | 0.667 |
| Mean confidence | 0.792 |
| Expected calibration error | 0.425 |
| Failure taxonomy | 3 calibrated_success, 2 overconfident_error, 1 underconfident_success |

The stub was *designed* to be overconfident on hard items, and the report
shows it: the 0.8–1.0 confidence bin has mean confidence 0.912 but mean
accuracy 0.500. That is the calibration gap this tool exists to surface.

Test suite: 13 unittest cases, all passing
(`python3 -m unittest discover -s tests`).

## Limitations

- The built-in graders are **keyword heuristics**, not judges of reasoning
  quality. Scores measure rubric-term coverage — a lower bound on whether
  the answer engaged the right concepts, not whether its reasoning is valid.
- Calibration bins are noisy with few cases per bin; treat ECE as a rough
  summary, not a precise measurement.
- The harness trusts the model's self-reported confidence; it cannot detect
  a model that reports calibrated-looking numbers strategically.
- No statistical significance testing is included; with small suites,
  apparent domain differences may be noise.

## Usage

```python
from reasoning_eval_harness import (
    EvalHarness, ModelResponse, RubricCriterion, TaskCase,
)

harness = EvalHarness()
harness.register_suite("my-suite", [
    TaskCase(
        case_id="t1",
        prompt="Explain the cobra effect.",
        gold_answer="paying per dead cobra incentivized breeding",
        domain="economics",
        tags=["perverse-incentives"],
        rubric=[RubricCriterion("names breeding",
                                ["breeding", "incentivized"], 1.0)],
    ),
])

def my_model(prompt: str) -> ModelResponse:
    return ModelResponse("paying per dead cobra incentivized breeding", 0.8)

report = harness.run(my_model, "my-suite")
print(report.summary())
harness.save_report(report, "report.json")
```

## Layout

- `src/reasoning_eval_harness/` — the package (stdlib only)
- `tests/` — unittest suite
- `examples/demo.py` — worked demo producing `demo-report.json`

## License

MIT — see `LICENSE`.
