"""Number perturbation for TTS experiments."""

import random
import re

NUMBER_PATTERN = re.compile(
    r"(?<![a-zA-Z_])"
    r"(-?\d+(?:\.\d+)?)"
    r"(?![a-zA-Z_])"
)

_OFFSETS = [-3, -2, -1, 1, 2, 3]


def has_numbers(text: str) -> bool:
    """Check whether text contains any numeric values."""
    return bool(NUMBER_PATTERN.search(text))


def perturb_numbers(text: str, rng: random.Random | None = None) -> str:
    """Perturb numerical values in text with small integer offsets.
    
    Follows Appendix A of the paper: "add small random offsets (chosen from
    [-3, -2, -1, 1, 2, 3]) to the numbers in a reasoning step."
    """
    if rng is None:
        rng = random.Random()

    def _perturb_match(match: re.Match) -> str:
        original = match.group(1)
        try:
            val = float(original)
        except ValueError:
            return original

        offset = rng.choice(_OFFSETS)
        new_val = val + offset

        if "." not in original:
            return str(int(new_val))
        else:
            decimals = len(original.split(".")[1])
            return f"{new_val:.{decimals}f}"

    return NUMBER_PATTERN.sub(_perturb_match, text)
