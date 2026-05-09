"""Configuration for the Reasoning Honesty Lab backend."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Tinker API
TINKER_API_KEY = os.environ.get("TINKER_API_KEY", "")

# Model configuration
BASE_MODEL = os.environ.get("BASE_MODEL", "deepseek-ai/DeepSeek-V3.1")
RENDERER_NAME = os.environ.get("RENDERER_NAME", "deepseekv3_thinking")

# TTS thresholds
TTS_HIGH_THRESHOLD = 0.7
TTS_DECORATIVE_THRESHOLD = 0.005

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BASELINE_DIR = DATA_DIR / "baseline"
RUNS_DIR = DATA_DIR / "runs"

# Ensure directories exist
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
