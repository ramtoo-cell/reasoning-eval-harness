"""Demo: build a small suite, run a stub model, print the report.

Run:  python examples/demo.py        (from the repo root)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reasoning_eval_harness import (
    EvalHarness,
    ModelResponse,
    RubricCriterion,
    TaskCase,
)


def build_suite():
    return [
        TaskCase(
            "demo-1", "What is 2 + 2?", "4",
            domain="math", tags=["arithmetic"],
            rubric=[RubricCriterion("says four", ["4"], 1.0)],
        ),
        TaskCase(
            "demo-2", "Name a primary color.", "red",
            domain="general", tags=["facts"],
            rubric=[RubricCriterion("names red", ["red"], 1.0)],
        ),
        TaskCase(
            "demo-3", "What is the capital of France?", "Paris",
            domain="general", tags=["facts"],
            rubric=[RubricCriterion("says Paris", ["paris"], 1.0)],
        ),
        TaskCase(
            "demo-4", "Explain opportunity cost in one sentence.",
            "the value of the best forgone alternative",
            domain="economics", tags=["opportunity-cost"],
            rubric=[
                RubricCriterion("names forgone alternative",
                                ["forgone", "alternative"], 1.0),
            ],
        ),
        TaskCase(
            "demo-5", "What is 17 * 23?", "391",
            domain="math", tags=["arithmetic"],
            rubric=[RubricCriterion("says 391", ["391"], 1.0)],
        ),
        TaskCase(
            "demo-6", "Is a promise binding if keeping it harms others?",
            "duties can be overridden when they cause harm",
            domain="philosophy", tags=["deontology"],
            rubric=[
                RubricCriterion("mentions override",
                                ["overridden", "harm"], 1.0),
            ],
        ),
    ]


# A deliberately imperfect stub: right on easy items, wrong and
# overconfident on hard ones, underconfident on one easy one.
STUB = {
    "demo-1": ("4", 0.95),
    "demo-2": ("red", 0.95),
    "demo-3": ("London", 0.85),   # wrong, confident
    "demo-4": ("the value of the best forgone alternative", 0.4),  # right, unsure
    "demo-5": ("400", 0.9),       # wrong, confident
    "demo-6": ("duties can be overridden when they cause harm", 0.7),
}

PROMPT_TO_ID = {c.prompt: c.case_id for c in build_suite()}


def model(prompt):
    case_id = PROMPT_TO_ID[prompt]
    answer, conf = STUB[case_id]
    return ModelResponse(answer, conf)


def main():
    harness = EvalHarness()
    harness.register_suite("demo", build_suite())
    report = harness.run(model, "demo")
    print(report.summary())
    out = os.path.join(os.path.dirname(__file__), "..", "demo-report.json")
    harness.save_report(report, os.path.abspath(out))
    print("\nSaved JSON report to demo-report.json")


if __name__ == "__main__":
    main()
