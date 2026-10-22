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
