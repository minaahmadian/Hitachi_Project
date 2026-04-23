"""
pytest conftest — make the project root importable so ``from core.enums …``
works when tests are run from anywhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Keep tests LLM-independent: never hit Groq from a pytest run.
os.environ.setdefault("GROQ_API_KEY", "")
