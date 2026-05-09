"""Generate mock baseline analysis for testing comparison."""

import json
import random
from datetime import datetime

import config
from cache.store import write_json


def generate_mock_baseline():
    """Generate realistic baseline analysis (worse than trained model)."""
    with open(config.DATA_DIR / "examples.json") as f:
        problems = json.load(f)

    analyses = []
    for prob in problems:
        # Baseline: more steps, lower TTS, more decorative
        n_steps = random.randint(20, 35)
        analyses.append({
            "id": prob["id"],
            "question": prob["question"],
            "answer": prob["answer"],
            "model_correct": random.random() > 0.3,  # 70% accuracy
            "n_steps": n_steps,
            "mean_tts": round(random.uniform(0.05, 0.12), 4),  # Lower TTS
            "frac_high_tts": round(random.uniform(0.03, 0.08), 4),
            "frac_decorative": round(random.uniform(0.3, 0.5), 4),  # More decorative
            "n_self_verification": random.randint(2, 5),
            "n_sv_decorative": random.randint(1, 3),
            "per_step_tts": [round(random.uniform(0.001, 0.25), 4) for _ in range(n_steps)],
        })

    baseline_data = {
        "model": config.BASE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "n_problems": len(problems),
        "analyses": analyses,
    }
    write_json(config.BASELINE_DIR / "baseline_analysis.json", baseline_data)
    print(f"Baseline analysis generated: {config.BASELINE_DIR / 'baseline_analysis.json'}")
    return baseline_data


if __name__ == "__main__":
    generate_mock_baseline()
