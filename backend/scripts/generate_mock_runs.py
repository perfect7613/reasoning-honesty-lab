"""Generate mock training run data for testing endpoints."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import config
from cache.store import write_json


def generate_mock_training_run(run_id: str, problems: list[dict]):
    """Generate realistic mock training data."""
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    num_steps = 50
    start_time = datetime.now() - timedelta(hours=2)

    # Config
    config_data = {
        "run_id": run_id,
        "model": config.BASE_MODEL,
        "num_steps": num_steps,
        "group_size": 8,
        "learning_rate": 1e-5,
        "lora_rank": 32,
        "started_at": start_time.isoformat(),
        "problems": [p["id"] for p in problems],
    }
    write_json(run_dir / "config.json", config_data)

    # Metrics over time (improving trend)
    metrics = []
    logs = []
    for step in range(1, num_steps + 1):
        # Simulate improvement over time
        progress = step / num_steps
        accuracy = 0.5 + 0.4 * progress + random.uniform(-0.05, 0.05)
        mean_tts = 0.05 + 0.15 * progress + random.uniform(-0.02, 0.02)
        mean_steps = 25 - 15 * progress + random.uniform(-2, 2)
        decorative = 0.5 - 0.4 * progress + random.uniform(-0.05, 0.05)
        reward = 0.4 + 0.4 * progress + random.uniform(-0.05, 0.05)

        metrics.append({
            "step": step,
            "mean_reward": round(max(0, reward), 4),
            "accuracy": round(min(1.0, max(0, accuracy)), 4),
            "mean_tts": round(max(0, mean_tts), 4),
            "mean_steps": round(max(5, mean_steps), 1),
            "mean_decorative": round(max(0, min(1.0, decorative)), 4),
            "timestamp": (start_time + timedelta(minutes=step * 3)).isoformat(),
        })

        logs.append(
            f"step {step}: reward={reward:.4f}, acc={accuracy:.2f}, "
            f"tts={mean_tts:.4f}, steps={mean_steps:.1f}, decorative={decorative:.2f}"
        )

    # Write metrics.jsonl
    with open(run_dir / "metrics.jsonl", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

    # Status
    final_metrics = metrics[-1]
    status = {
        "run_id": run_id,
        "status": "completed",
        "current_step": num_steps,
        "total_steps": num_steps,
        "latest_metrics": final_metrics,
        "logs": logs,
        "completed_at": (start_time + timedelta(minutes=num_steps * 3)).isoformat(),
    }
    write_json(run_dir / "status.json", status)

    # Checkpoint
    checkpoint_data = {
        "sampler_path": f"tinker://mock/{run_id}/sampler",
        "state_path": f"tinker://mock/{run_id}/state",
        "training_run_id": f"mock-run-{run_id}",
    }
    write_json(run_dir / "checkpoint.json", checkpoint_data)

    # Improved analysis (better than baseline)
    analyses = []
    for prob in problems:
        # Simulate improved reasoning: fewer steps, higher TTS, less decorative
        analyses.append({
            "id": prob["id"],
            "question": prob["question"][:100],
            "answer": prob["answer"],
            "model_correct": True,
            "n_steps": random.randint(8, 15),  # Fewer steps than baseline
            "mean_tts": round(random.uniform(0.15, 0.25), 4),  # Higher TTS
            "frac_high_tts": round(random.uniform(0.15, 0.25), 4),
            "frac_decorative": round(random.uniform(0.05, 0.15), 4),  # Less decorative
            "n_self_verification": random.randint(0, 1),
            "n_sv_decorative": 0,
            "per_step_tts": [round(random.uniform(0.01, 0.3), 4) for _ in range(random.randint(8, 15))],
        })

    improved_data = {
        "model": config.BASE_MODEL,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "analyses": analyses,
    }
    write_json(run_dir / "improved_analysis.json", improved_data)

    # Run result
    run_result = {
        "run_id": run_id,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "num_steps": num_steps,
        "final_accuracy": final_metrics["accuracy"],
        "final_mean_tts": final_metrics["mean_tts"],
        "final_mean_steps": final_metrics["mean_steps"],
        "final_decorative": final_metrics["mean_decorative"],
        "checkpoint": checkpoint_data,
    }
    write_json(run_dir / "run_result.json", run_result)

    print(f"Mock training run generated: {run_dir}")
    return run_result


if __name__ == "__main__":
    with open(config.DATA_DIR / "examples.json") as f:
        problems = json.load(f)

    # Generate 2 mock runs
    for i in range(2):
        run_id = f"mock_run_{datetime.now().strftime('%Y%m%d')}_{i+1}"
        result = generate_mock_training_run(run_id, problems[:10])
        print(f"  - {run_id}: acc={result['final_accuracy']:.2f}, tts={result['final_mean_tts']:.4f}")
