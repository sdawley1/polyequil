"""Reusable Hypothesis strategies."""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from polyequil.box import Box
from polyequil.topology import ChainTopology, linear_chains

finite = st.floats(
    -1e3, 1e3, allow_nan=False, allow_infinity=False, allow_subnormal=False
)
lengths = st.floats(0.5, 50.0, allow_nan=False, allow_infinity=False)


@st.composite
def boxes(draw: st.DrawFn) -> Box:
    lo = draw(hnp.arrays(np.float64, 3, elements=st.floats(-10, 10)))
    size = draw(hnp.arrays(np.float64, 3, elements=lengths))
    return Box(lo=lo, lengths=size)


@st.composite
def topologies(
    draw: st.DrawFn, max_chains: int = 6, max_length: int = 12
) -> ChainTopology:
    sizes = draw(st.lists(st.integers(1, max_length), min_size=1, max_size=max_chains))
    return linear_chains(sizes)


def points(n: int) -> st.SearchStrategy[np.ndarray]:
    return hnp.arrays(np.float64, (n, 3), elements=finite)
