"""
Argument checks shared by the observables.

Andrea Aude
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- local -----
from polyequil.errors import ValidationError

# ============================================================
# PUBLIC API
# ============================================================


def check_threads(threads: int) -> None:
    """Reject a non-positive thread count."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValidationError(f"threads must be a positive int, got {threads!r}")
