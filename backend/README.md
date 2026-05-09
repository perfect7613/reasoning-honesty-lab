# Reasoning Honesty Lab - Backend

FastAPI backend for TTS-powered reasoning analysis and RL fine-tuning.

## Setup

```bash
cd backend
uv sync
```

Create a `.env` file:
```
TINKER_API_KEY=your-key-here
BASE_MODEL=deepseek-ai/DeepSeek-V3.1
RENDERER_NAME=deepseekv3_thinking
```

## Run

```bash
uv run uvicorn main:app --reload --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/problems` | GET | List 20 curated MATH-500 problems |
| `/analyze` | POST | Analyze a problem (baseline cached or live TTS) |
| `/train/start` | POST | Start a background training job |
| `/train/status/{run_id}` | GET | Get training job status and logs |
| `/runs` | GET | List completed training runs |
| `/runs/{run_id}` | GET | Get before/after comparison for a run |
| `/export/{analysis_id}` | GET | Export reasoning receipt JSON |

## Tests

```bash
uv run pytest tests/ -v
```

## Precompute Baseline

```bash
uv run python scripts/precompute_baseline.py
```

## Project Structure

```
backend/
├── main.py              # FastAPI app
├── config.py            # Configuration (.env support)
├── tts/
│   ├── segmenter.py     # CoT step segmentation
│   ├── perturb.py       # Number perturbation
│   ├── scorer.py        # TTS computation
│   └── client.py        # Tinker client wrapper
├── rl/
│   ├── reward.py        # TTS-aware reward function
│   ├── env.py           # Math environment
│   └── train.py         # Training loop
├── cache/
│   └── store.py         # JSON read/write utilities
├── scripts/
│   └── precompute_baseline.py
└── tests/
    ├── test_tts.py      # Segmenter, perturb, scorer tests
    └── test_reward.py   # Reward function tests
```
