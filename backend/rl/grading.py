"""Answer extraction and exact-match grading for math rollouts."""

import re


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last simple \\boxed{} answer from model output."""
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer strings for exact-match math grading."""
    return (
        answer.strip()
        .replace(" ", "")
        .replace(",", "")
        .replace("$", "")
        .lower()
    )


def grade_response_exact(response_text: str, ground_truth: str) -> bool:
    """Return True only when the final boxed answer exactly matches ground truth."""
    predicted = extract_boxed_answer(response_text)
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(ground_truth)
