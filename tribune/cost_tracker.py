"""Re-export cost_tracker from top level for package imports."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cost_tracker import *  # noqa: F401, F403, E402
