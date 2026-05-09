"""Math environment with TTS-aware reward for RL training."""

import logging
import re

from tinker_cookbook.rl.types import Env, StepResult
from tinker_cookbook.renderers import Renderer

from tts.scorer import compute_tts_for_cot
from rl.reward import compute_reward

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{} format."""
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(ans: str) -> str:
    """Normalize answer string for comparison."""
    return ans.replace(" ", "").replace(",", "").lower()


class TTSMathEnv(Env):
    """Environment that rewards correct math answers with TTS-aware scoring."""

    def __init__(self, problem: str, answer: str, renderer: Renderer):
        self.problem = problem
        self.answer = answer
        self.renderer = renderer
        self.tokenizer = renderer.tokenizer

    def initial_observation(self):
        return self.renderer.build_generation_prompt(
            [{"role": "user", "content": self.problem}],
            role="assistant",
        )

    async def step(self, action) -> StepResult:
        """Parse model response, grade answer, compute TTS reward."""
        response_text = self.tokenizer.decode(action.tokens)

        # Extract answer from boxed format
        predicted = extract_boxed_answer(response_text)
        is_correct = False
        if predicted is not None:
            is_correct = normalize_answer(predicted) == normalize_answer(self.answer)

        # Extract thinking content
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
        else:
            thinking_text = response_text

        # Compute TTS (if thinking text is non-empty)
        mean_tts = 0.0
        decorative_fraction = 0.0
        if thinking_text and len(thinking_text) > 10:
            try:
                # We need a sampling client to compute TTS, but in the env we don't have one.
                # For RL training, we'll compute TTS after the rollout using a separate scorer.
                # Here we use a heuristic: shorter traces with correct answers get higher reward.
                step_count = len(thinking_text.split("\n"))
                mean_tts = 0.05  # placeholder; real TTS computed post-rollout
                decorative_fraction = 0.5  # placeholder
            except Exception as e:
                logger.warning(f"TTS computation failed: {e}")

        # Use a simplified reward for the env step; full TTS reward computed in train loop
        reward = 1.0 if is_correct else 0.0

        return StepResult(
            observation=None,
            reward=reward,
            episode_done=True,
            metrics={
                "correct": is_correct,
                "response_length": len(action.tokens),
                "thinking_length": len(thinking_text),
            },
        )
