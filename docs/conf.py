"""
Sphinx configuration for the polyequil documentation.

Pages are Markdown (MyST); the API reference is generated from the NumPy-style
docstrings by autodoc and napoleon. Build with
``uv run --group docs sphinx-build -W docs docs/_build/html``.

Hong-Ye Wang
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from importlib.metadata import version as _version

# ============================================================
# PROJECT
# ============================================================

project = "polyequil"
author = "Andrea Aude, Sam Dawley, Hong-Ye Wang"
copyright = f"2026, {author}"
release = _version("polyequil")

# ============================================================
# EXTENSIONS
# ============================================================

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

exclude_patterns = ["_build"]

# ============================================================
# OUTPUT
# ============================================================

html_theme = "furo"
html_title = f"polyequil {release}"
