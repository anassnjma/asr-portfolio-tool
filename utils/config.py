"""Configuration and environment setup."""

import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Gemini
GEMINI_MODEL = "gemini-1.5-flash"

# Simulation defaults
SIMULATION_PATHS = 100_000
SIMULATION_YEARS = 15
TRADING_DAYS_PER_YEAR = 252
BOOTSTRAP_BLOCK_SIZE = 21  # ~1 month of trading days


def get_gemini_api_key() -> str:
    """Get API key from .env file."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. To fix this:\n"
            "  1. Get a free key at https://aistudio.google.com/\n"
            "  2. Copy .env.example to .env and paste your key\n\n"
            "  Or run with --no-llm to use direct commands instead."
        )
    return key
