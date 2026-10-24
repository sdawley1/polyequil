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

# ----- reciprocal space and structure factors -----
FOURIER_KERNEL: str = "exp(-i k.r)"
RECIPROCAL_VECTORS: str = (
    "k = 2 pi n / L, n integer; half-space (k and -k counted once)"
)
SHELLS: str = "modes grouped by exactly equal |k|; shell value = mean over its modes"
SHELL_RELATIVE_TOLERANCE: float = 1e-10
S_K_NORMALIZATION: str = "|sum_j exp(-i k.r_j)|^2 / N over monomers; k = 0 excluded"

# ----- orientation -----
SEGMENT_DEFINITION: str = (
    "unit vector from bead i to bead i + n_s of the same chain, for every such i "
    "(overlapping), located at the midpoint; the aligning field must use this set"
)
Q_BASIS: str = (
    "orthonormal traceless symmetric basis: (xx-yy)/sqrt2, (2zz-xx-yy)/sqrt6, "
    "sqrt2 xy, sqrt2 xz, sqrt2 yz; so Q:Q = sum_a q_a^2"
)
S_Q_PREFACTOR: float = 3.0 / 10.0
S_Q_NORMALIZATION: str = (
    "(3/10) sum_a |sum_s q_a(u_s) exp(-i k.r_s)|^2 / N_seg; equals "
    "<(sum_s P2(u_s.n))^2>/N_seg averaged over directions n; k = 0 included"
)

# ----- small-k extrapolation -----
SMALL_K_FORM: str = "S(k) = S0 + a k^2, weighted least squares over 0 < k <= k_fit"
SMALL_K_WEIGHTS: str = "shell mode counts (geometry only, so the fit is linear in S)"

# ----- equilibrium indices (k_B = 1, LJ units) -----
DENSITY_INDEX: str = "X_rho = S0 / (rho T kappa_T) - 1"
ORIENTATIONAL_INDEX: str = "X_Q = S_Q(0) / (T chi) - 1"
INDEX_ERROR_PROPAGATION: str = (
    "first order, independent errors in numerator and response"
)
