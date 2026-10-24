"""Module-level (picklable) observables used by the ensemble tests."""

import os

import numpy as np

from polyequil.configuration import ChainConfiguration
from polyequil.results import ObservableResult, SeparableAxes
from polyequil.trajectory import WindowData


def axes(**named: object) -> SeparableAxes:
    """Build SeparableAxes from keyword arrays (test shorthand)."""
    return SeparableAxes(
        tuple((k, np.asarray(v, dtype=float)) for k, v in named.items())
    )


def bead_count(config: ChainConfiguration, *, threads: int = 1) -> ObservableResult:
    return ObservableResult(np.asarray(float(config.n_beads)))


def mean_unwrapped(config: ChainConfiguration, *, threads: int = 1) -> ObservableResult:
    return ObservableResult(
        config.unwrapped().mean(axis=0), axes(component=[0.0, 1.0, 2.0])
    )


def fails_at_step(
    config: ChainConfiguration, *, threads: int = 1, step: int = 0
) -> ObservableResult:
    if config.step == step:
        raise RuntimeError(f"boom at step {step}")
    return ObservableResult(np.asarray(1.0))


def exits_at_step(
    config: ChainConfiguration, *, threads: int = 1, step: int = 0
) -> ObservableResult:
    if config.step == step:
        os._exit(1)
    return ObservableResult(np.asarray(1.0))


def shape_changes(config: ChainConfiguration, *, threads: int = 1) -> ObservableResult:
    size = 1 + int(config.step > 0)
    return ObservableResult(np.zeros(size), axes(k=np.arange(size)))


def gaussian(
    config: ChainConfiguration, *, rng: np.random.Generator, threads: int = 1
) -> ObservableResult:
    return ObservableResult(np.asarray(rng.normal()))


def window_span(window: WindowData, *, threads: int = 1) -> ObservableResult:
    values = np.array([window.n_frames, window.times[-1] - window.times[0]])
    return ObservableResult(values, axes(quantity=[0.0, 1.0]))


def missing_threads(config: ChainConfiguration) -> ObservableResult:
    return ObservableResult(np.asarray(0.0))


def positional_threads(config: ChainConfiguration, threads: int, /) -> ObservableResult:
    return ObservableResult(np.asarray(0.0))


def square(value: float) -> float:
    return value * value


def raise_on_three(value: int) -> int:
    if value == 3:
        raise ValueError("three is not allowed")
    return value


def exit_on_four(value: int) -> int:
    if value == 4:
        os._exit(1)
    return value


def blas_thread_count(value: int) -> int:
    """Return the largest native thread-pool size visible to this task."""
    from threadpoolctl import threadpool_info

    return max((pool["num_threads"] for pool in threadpool_info()), default=1)


def quadratic_shells(
    config: ChainConfiguration,
    *,
    threads: int = 1,
    k_max: float,
    s0: float,
    a: float,
    rng: np.random.Generator | None = None,
) -> ObservableResult:
    """S(k) = s0 + a k^2 on the box's shells; with rng, each shell is the mean of
    its modes, each exponentially distributed (the ideal per-mode noise)."""
    from polyequil.reciprocal import reciprocal_shells

    shells = reciprocal_shells(config.box, k_max, include_zero=False)
    mean = s0 + a * shells.shell_k**2
    counts = shells.shell_counts
    values = mean if rng is None else rng.gamma(counts, mean / counts)
    return ObservableResult(
        values,
        SeparableAxes((("k", shells.shell_k),)),
        auxiliary={"shell_counts": counts},
    )
