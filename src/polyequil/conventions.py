"""
Single home for the physical and storage conventions of polyequil.

Every module imports these names instead of restating them (AGENTS.md rule 3).

Conventions
-----------
- Units: Lennard-Jones reduced units (sigma, epsilon, tau, bead mass 1).
- Boxes are orthorhombic, described by a lower corner ``lo`` and edge lengths.
- Positions are stored wrapped into ``[lo, lo + L)`` together with integer
  image flags; the unwrapped position is ``x + images * L``.
- Minimum image: ``d - L * floor(d / L + 1/2)``, so a separation of exactly
  ``+L/2`` maps to ``-L/2`` (heteromorph's tie rule).
- Time is in tau; ``t = 0`` is wherever the producing protocol put it, recorded
  in the ``time_origin`` file attribute.
- Topology indices are zero-based; chains are linear and their beads are
  contiguous and ordered along the backbone.

Sam Dawley
10/2026
"""

# ============================================================
# CONSTANTS
# ============================================================

SPATIAL_DIMENSION: int = 3
UNITS: str = "lj"

SCHEMA: str = "polyequil-1"
READABLE_SCHEMAS: tuple[str, ...] = ("polyequil-1", "kg_template-1")

IMAGES_FROM_DUMP: str = "dump"
IMAGES_FROM_BONDS: str = "reconstructed_from_bonds"
IMAGES_SOURCES: tuple[str, ...] = (IMAGES_FROM_DUMP, IMAGES_FROM_BONDS)

POSITIONS_CONVENTION: str = "wrapped into [lo, hi); unwrapped = x + image * L"

FRAME_CHUNK_COMPRESSION: str = "gzip"
FRAME_CHUNK_COMPRESSION_LEVEL: int = 4
