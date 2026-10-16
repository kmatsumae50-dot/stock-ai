from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Build Version / Cache Buster for Streamlit Cloud File Watcher
# BUILD_TIMESTAMP: 2026-10-16T16:05:00+09:00 - v2.2.0-discord-safe-import
APP_BUILD_VERSION = "2026.10.16.v3"

# Ensure src/ is on the Python module search path for Streamlit Cloud and container runtimes
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Force hot-reloading of UI/notification modules if previously imported in runtime
for mod_name in [
    "stock_ai.cloud.notifications",
    "stock_ai.app.main",
]:
    if mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

import stock_ai.app.main as main_mod  # noqa: E402

main = main_mod.main

if __name__ == "__main__" or "streamlit" in sys.modules:
    main()

