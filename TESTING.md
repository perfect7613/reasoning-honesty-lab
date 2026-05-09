# Manual Testing Guide - Reasoning Honesty Lab

## Prerequisites

Make sure you're in the project root:
```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl
```

## 1. Backend Unit Tests

Run all backend tests:
```bash
cd backend
uv run pytest tests/ -v
```

Expected: **24 tests passing**

Run individual test files:
```bash
uv run pytest tests/test_tts.py -v
uv run pytest tests/test_reward.py -v
uv run pytest tests/test_issues.py -v
```

## 2. Start Backend Server

Terminal 1 - Start the API server:
```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl/backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 3. Test API Endpoints (with curl)

Open a new terminal (Terminal 2) and run these commands:

### 3.1 Health Check
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Expected:
```json
{
    "status": "ok",
    "model": "deepseek-ai/DeepSeek-V3.1"
}
```

### 3.2 List Problems
```bash
curl -s http://localhost:8000/problems | python3 -m json.tool | head -30
```

Expected: Array of 20 problems with id, question, answer

### 3.3 Analyze Problem (Baseline - Cached)
```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"problem_id": "math_001", "mode": "baseline"}' | python3 -m json.tool
```

Expected: TTS analysis with n_steps, mean_tts, per_step_tts, etc.

### 3.4 Start Training Job
```bash
curl -s -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{"problem_ids": ["math_001", "math_002"], "num_steps": 5, "group_size": 2}' | python3 -m json.tool
```

Expected:
```json
{
    "run_id": "run_2026...",
    "status": "started"
}
```

### 3.5 Check Training Status
```bash
curl -s http://localhost:8000/train/status/run_2026... | python3 -m json.tool
```

Note: Replace `run_2026...` with the actual run_id from step 3.4

Expected: status, current_step, logs, latest_metrics

### 3.6 List Completed Runs
```bash
curl -s http://localhost:8000/runs | python3 -m json.tool
```

Expected: Array of training runs with metrics

### 3.7 Get Run Comparison
```bash
curl -s http://localhost:8000/runs/mock_run_20260509_1 | python3 -m json.tool | head -50
```

Expected: baseline + improved analyses for side-by-side comparison

### 3.8 Export Analysis
```bash
curl -s http://localhost:8000/export/math_001 | python3 -m json.tool | head -20
```

Expected: JSON reasoning receipt

## 4. Live TTS Analysis (Costs API Credits)

⚠️ **Warning**: This makes real API calls to Tinker and costs credits.

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2+2?", "answer": "4", "mode": "live"}' | python3 -m json.tool
```

Takes 2-3 minutes. Generates CoT and computes TTS.

## 5. Run Training Script (Costs API Credits)

⚠️ **Warning**: This trains a real model and costs significant credits.

```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl/backend
PYTHONPATH=/Users/ameymuke/Documents/Builds/Other/rsl/backend \
uv run python scripts/test_training.py
```

This runs a mini training job (2 problems, 2 steps) to verify the full loop works.

## 6. Generate Mock Data (No API Calls)

If you want fresh mock data without spending credits:

```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl/backend
PYTHONPATH=/Users/ameymuke/Documents/Builds/Other/rsl/backend \
uv run python scripts/generate_mock_baseline.py

PYTHONPATH=/Users/ameymuke/Documents/Builds/Other/rsl/backend \
uv run python scripts/generate_mock_runs.py
```

## 7. Frontend Setup

### 7.1 Install Dependencies
```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl/frontend
npm install
```

If npm fails due to SSL:
```bash
NODE_TLS_REJECT_UNAUTHORIZED=0 npm install
```

### 7.2 Start Dev Server
```bash
npm run dev
```

Opens at http://localhost:3000

### 7.3 Build for Production
```bash
npm run build
```

## 8. Full Integration Test

1. Start backend: `uv run uvicorn main:app --host 0.0.0.0 --port 8000`
2. Start frontend: `npm run dev` (in another terminal)
3. Open http://localhost:3000
4. Select a problem from dropdown
5. See baseline analysis with issues banner + step heatmap
6. Click "FIX REASONING"
7. See training panel open with logs
8. After 5 seconds, click "HISTORY" tab
9. Click a completed run
10. See before/after comparison

## 9. Test Specific Backend Components

### Test TTS on a single problem:
```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl/backend
PYTHONPATH=/Users/ameymuke/Documents/Builds/Other/rsl/backend \
uv run python scripts/test_tts_reward.py
```

### Test reward function:
```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl/backend
uv run python -c "
from rl.reward import compute_reward
r1 = compute_reward(True, 0.15, 0.0)
r2 = compute_reward(True, 0.05, 0.5)
print(f'Good reasoning: {r1:.4f}')
print(f'Bad reasoning:  {r2:.4f}')
print(f'Difference:     {r1-r2:.4f}')
"
```

### Test issue detection:
```bash
cd /Users/ameymuke/Documents/Builds/Other/rsl/backend
uv run python -c "
from rl.issues import detect_issues
from tts.scorer import StepTTS

steps = [
    StepTTS(0, ' filler ', 0.001, 0,0,0,0, False),
    StepTTS(1, 'Wait...', 0.0001, 0,0,0,0, True),
    StepTTS(2, 'calc', 0.85, 0,0,0,0, False),
]
issues = detect_issues(steps)
print('Issues found:')
for i in issues:
    print(f'  - {i}')
"
```

## 10. Debug Commands

### Check server logs:
```bash
cat /tmp/uvicorn.log
```

### Check training run files:
```bash
ls -la /Users/ameymuke/Documents/Builds/Other/rsl/data/runs/
```

### Check baseline:
```bash
cat /Users/ameymuke/Documents/Builds/Other/rsl/data/baseline/baseline_analysis.json | head -50
```

### Check if port 8000 is in use:
```bash
lsof -i :8000
```

### Kill server if stuck:
```bash
pkill -f "uvicorn main:app"
```

## Expected Results Summary

| Test | Expected Result |
|------|----------------|
| Unit tests | 24 passing |
| Health check | `{"status": "ok"}` |
| List problems | 20 problems |
| Baseline analyze | Cached JSON with TTS |
| Training start | Returns run_id |
| Training status | Running/completed |
| List runs | Array of runs |
| Run comparison | baseline + improved |
| Export | JSON receipt |
| Frontend | Dark UI, interactive |
