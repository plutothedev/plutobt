"""Root conftest: add repo root to sys.path so 'examples' is importable in tests."""
import sys
from pathlib import Path

# The repo root (where examples/ lives) must be on sys.path for tests that import
# from examples.run_demo.  src/ is already on the path via the editable install;
# the root is not, so we add it here once.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
