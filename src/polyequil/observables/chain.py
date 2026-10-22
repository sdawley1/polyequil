"""
Single-chain observables of one frame.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil.configuration import ChainConfiguration
from polyequil.errors import ValidationError
from polyequil.results import ObservableResult

# ============================================================
# PUBLIC API
# ============================================================


def radius_of_gyration(
    configuration: ChainConfiguration, *, threads: int = 1
) -> ObservableResult:
    """Return the chain-averaged squared radius of gyration, in sigma^2.

    Parameters
    ----------
    configuration : ChainConfiguration
        One frame; chains are unwrapped with the stored image flags.
    threads : int
        Reserved by the execution layer; this observable is single-threaded.

    Returns
    -------
    ObservableResult
        Scalar ``<R_g^2>``, averaged with equal weight over chains.
    """
    if threads < 1:
        raise ValidationError(f"threads must be >= 1, got {threads}")
    topology = configuration.topology
    rg2 = per_chain_rg2(configuration.unwrapped(), topology.chain_id)
    return ObservableResult(
        values=np.asarray(rg2.mean()),
        convention={"quantity": "mean squared radius of gyration", "units": "sigma^2"},
    )


def per_chain_rg2(
    positions: NDArray[np.float64], chain_id: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return R_g^2 of each chain from unwrapped positions of shape (n, 3)."""
    if positions.shape != (chain_id.shape[0], 3):
        raise ValidationError(f"positions shape {positions.shape} != (n, 3)")
    counts = np.bincount(chain_id).astype(np.float64)
    assert np.all(counts > 0), "chain ids are contiguous from zero"
    centres = (
        np.stack(
            [np.bincount(chain_id, weights=positions[:, k]) for k in range(3)], axis=1
        )
        / counts[:, None]
    )
    squared = np.sum((positions - centres[chain_id]) ** 2, axis=1)
    rg2: NDArray[np.float64] = np.bincount(chain_id, weights=squared) / counts
    return rg2
