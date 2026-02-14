"""Configuration for SDK integration tests.

Set these environment variables or add to .env file:
    MLPAL_API_KEY=mlpal_sk_xxx
    MLPAL_BASE_URL=https://models.mlpal.ai  (optional, defaults to this)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Base URL (defaults to production)
BASE_URL = os.environ.get("MLPAL_BASE_URL", "https://models.mlpal.ai")

# API key
API_KEY = os.environ.get("MLPAL_API_KEY", "")

if not API_KEY:
    print("WARNING: MLPAL_API_KEY not set. Tests will fail.")
    print("Set: export MLPAL_API_KEY=mlpal_sk_xxx")
