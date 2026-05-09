"""FastAPI backend for Reasoning Honesty Lab."""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from cache.store import read_json, write_json, list_dirs
from rl.train import run_training
from tts.client import generate_cot_and_compute_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TINKER_API_KEY", config.TINKER_API_KEY)


class AnalyzeRequest(BaseModel):
    problem_id: str | None = None
    question: str | None = None
    answer: str | None = None
    mode: str = "baseline"  # "baseline" or "live"


class TrainRequest(BaseModel):
    problem_ids: list[str] | None = None
    num_steps: int = 20
    group_size: int = 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Reasoning Honesty Lab backend...")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Reasoning Honesty Lab API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load problems at startup
_problems = []


@asynccontextmanager
async def load_problems():
    global _problems
    problems_path = config.DATA_DIR / "examples.json"
    if problems_path.exists():
        import json
        with open(problems_path) as f:
            _problems = json.load(f)
    yield


@app.get("/problems")
async def get_problems():
    """List all curated problems."""
    if not _problems:
        problems_path = config.DATA_DIR / "examples.json"
        if problems_path.exists():
            import json
            with open(problems_path) as f:
                return json.load(f)
        return []
    return _problems


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyze a problem: serve from cache or compute live TTS."""
    import tinker

    # Load problem if ID provided
    question = req.question
    answer = req.answer
    if req.problem_id and not question:
        problems_path = config.DATA_DIR / "examples.json"
        if problems_path.exists():
            import json
            with open(problems_path) as f:
                problems = json.load(f)
            for p in problems:
                if p["id"] == req.problem_id:
                    question = p["question"]
                    answer = p["answer"]
                    break

    if not question or not answer:
        return {"error": "Question and answer required"}

    # Check baseline cache
    if req.mode == "baseline":
        baseline_path = config.BASELINE_DIR / "baseline_analysis.json"
        baseline = read_json(baseline_path)
        if baseline:
            # Find the specific problem in the baseline
            for item in baseline.get("analyses", []):
                if item.get("question") == question or item.get("question", "").startswith(question[:50]):
                    item["id"] = req.problem_id or item.get("id", "")
                    return item

    # Live analysis
    logger.info(f"Running live analysis for: {question[:60]}...")
    service_client = tinker.ServiceClient()
    result = await generate_cot_and_compute_tts(
        service_client=service_client,
        model_name=config.BASE_MODEL,
        question=question,
        answer_str=answer,
        renderer_name=config.RENDERER_NAME,
    )
    
    # result is already a summary dict with cot_text included
    result["id"] = req.problem_id or ""
    return result


@app.post("/train/start")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start a training job in the background."""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load problems
    problems_path = config.DATA_DIR / "examples.json"
    import json
    with open(problems_path) as f:
        all_problems = json.load(f)

    if req.problem_ids:
        problems = [p for p in all_problems if p["id"] in req.problem_ids]
    else:
        problems = all_problems[:5]

    # Start background training
    background_tasks.add_task(
        run_training,
        run_id=run_id,
        problems=problems,
        num_steps=req.num_steps,
        group_size=req.group_size,
    )

    return {"run_id": run_id, "status": "started"}


@app.get("/train/status/{run_id}")
async def get_training_status(run_id: str):
    """Get status of a training run."""
    status_path = config.RUNS_DIR / run_id / "status.json"
    status = read_json(status_path)
    if not status:
        return {"run_id": run_id, "status": "not_found"}
    return status


@app.get("/runs")
async def list_runs():
    """List all completed training runs."""
    runs = []
    for run_dir in list_dirs(config.RUNS_DIR):
        run_result_path = run_dir / "run_result.json"
        run_result = read_json(run_result_path)
        if run_result:
            runs.append(run_result)
        else:
            # Fallback: check if status is completed
            status_path = run_dir / "status.json"
            status = read_json(status_path)
            if status and status.get("status") == "completed":
                runs.append({
                    "run_id": run_dir.name,
                    "status": "completed",
                    "timestamp": status.get("completed_at", ""),
                })

    # Sort by timestamp descending
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return runs


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get full before/after comparison for a run."""
    run_dir = config.RUNS_DIR / run_id

    # Load improved analysis
    improved_path = run_dir / "improved_analysis.json"
    improved = read_json(improved_path)

    # Load baseline for comparison
    baseline_path = config.BASELINE_DIR / "baseline_analysis.json"
    baseline = read_json(baseline_path)

    return {
        "run_id": run_id,
        "baseline": baseline,
        "improved": improved,
    }


@app.get("/export/{analysis_id}")
async def export_analysis(analysis_id: str):
    """Export a reasoning receipt as JSON."""
    # Try baseline first
    baseline_path = config.BASELINE_DIR / "baseline_analysis.json"
    baseline = read_json(baseline_path)
    if baseline:
        for item in baseline.get("analyses", []):
            if item.get("question", "").startswith(analysis_id) or item.get("id") == analysis_id:
                return item

    # Try runs
    for run_dir in list_dirs(config.RUNS_DIR):
        improved_path = run_dir / "improved_analysis.json"
        improved = read_json(improved_path)
        if improved:
            for item in improved.get("analyses", []):
                if item.get("question", "").startswith(analysis_id) or item.get("id") == analysis_id:
                    return item

    return {"error": "Analysis not found"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.BASE_MODEL}
