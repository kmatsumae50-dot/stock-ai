from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the Python module search path for Streamlit Cloud and container runtimes
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_ai.app.main import main  # noqa: E402

if __name__ == "__main__" or "streamlit" in sys.modules:
    main()
