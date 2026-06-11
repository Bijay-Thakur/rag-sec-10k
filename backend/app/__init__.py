"""FastAPI application package."""

from __future__ import annotations

import sys
import types

# Docker: package is imported as top-level `app` but code uses `backend.app.*`
if __name__ == "app":
    _backend = sys.modules.get("backend")
    if _backend is None:
        _backend = types.ModuleType("backend")
        sys.modules["backend"] = _backend
    sys.modules["backend.app"] = sys.modules[__name__]
