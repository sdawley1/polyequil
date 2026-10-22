"""
Time-resolved averaging of observables over independent trajectories.

Adapted from heteromorph's ``ensemble.py``. A :class:`TrajectoryEnsemble` holds
M independent replicas (lazy :class:`~polyequil.trajectory.Trajectory` views).
Evaluation creates one task per (replica, frame) or (replica, window), runs the
tasks under an :class:`~polyequil.execution.ExecutionPolicy`, and reduces each
time slice across replicas only. Frames of one trajectory are never averaged
together: they are correlated, and the reported standard error assumes
independent samples.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import contextlib
import inspect
import pickle
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil._metadata import freeze_mapping, thaw_metadata
from polyequil._randomness import (
    StreamRoot,
    child_generator,
    find_random_state,
    root_stream,
)
from polyequil.accumulators import Accumulator, SampleStore, WelfordAccumulator
from polyequil.errors import EnsembleError, ValidationError
from polyequil.execution import ExecutionPolicy, execute
from polyequil.results import (
    Coordinates,
    ObservableResult,
    frozen_values,
    results_compatible,
)
from polyequil.topology import ChainTopology
from polyequil.trajectory import FrameRef, Trajectory, TrajectoryWindow

# ============================================================
# CONSTANTS
# ============================================================

KINDS: tuple[str, ...] = ("frame", "window")
RESERVED_PARAMETERS: tuple[str, ...] = ("threads", "rng")
_BY_KEYWORD = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]
Observable = Callable[..., ObservableResult]
Payload = tuple[
    Any, Observable, Mapping[str, Any], int, StreamRoot | None, tuple[int, int]
]


@dataclass(frozen=True, slots=True, eq=False)
class TimeResolvedResult:
    """The ensemble mean of one observable at each time slice.

    ``kind`` is ``"frame"`` (``times`` has shape (T,)) or ``"window"``
    (``times`` holds ``[t_start, t_stop]`` rows, shape (T, 2)). ``mean`` has
    shape (T, *value_shape). With one replica there is no uncertainty and both
    deviations are ``None``; an unavailable uncertainty is never reported as
    zero. ``samples`` (T, M, *value_shape) is kept only on request.
    """

    kind: str
    times: FloatArray
    mean: NDArray[Any]
    standard_deviation: FloatArray | None
    standard_error: FloatArray | None
    replica_count: int
    coordinates: Coordinates
    metadata: Mapping[str, Any]
    samples: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValidationError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if isinstance(self.replica_count, bool) or self.replica_count < 1:
            raise ValidationError("replica_count must be a positive int")
        times = np.array(self.times, dtype=np.float64)
        if times.shape[1:] != ((2,) if self.kind == "window" else ()):
            raise ValidationError(f"times shape {times.shape} does not fit {self.kind}")
        times.setflags(write=False)
        mean = frozen_values(self.mean)
        if mean.shape[:1] != times.shape[:1]:
            raise ValidationError("mean must have one row per time slice")
        self.coordinates.validate_against(mean.shape[1:])
        _validate_uncertainty(self, mean.shape)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "mean", mean)
        if self.samples is not None:
            samples = frozen_values(self.samples)
            expected = (mean.shape[0], self.replica_count, *mean.shape[1:])
            if samples.shape != expected:
                raise ValidationError(f"samples shape {samples.shape} != {expected}")
            object.__setattr__(self, "samples", samples)
        metadata = freeze_mapping(self.metadata, role="metadata", allow_opaque=True)
        object.__setattr__(self, "metadata", metadata)

    @property
    def has_uncertainty(self) -> bool:
        """Return whether standard deviations and errors are available."""
        return self.standard_error is not None

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor so arrays and metadata stay frozen."""
        return (
            type(self),
            (
                self.kind,
                self.times,
                self.mean,
                self.standard_deviation,
                self.standard_error,
                self.replica_count,
                self.coordinates,
                thaw_metadata(self.metadata),
                self.samples,
            ),
        )


class TrajectoryEnsemble:
    """Independent replica trajectories of one system, evaluated slice by slice.

    Parameters
    ----------
    trajectories : Iterable[Trajectory | str | Path]
        One file per independent replica. Paths must be distinct, topologies
        identical, and any ``root_seed`` attributes distinct.
    """

    __slots__ = ("_trajectories",)

    def __init__(self, trajectories: Iterable[Trajectory | str | Path]) -> None:
        items = tuple(
            t if isinstance(t, Trajectory) else Trajectory(t) for t in trajectories
        )
        if not items:
            raise ValidationError("an ensemble needs at least one trajectory")
        _check_independent(items)
        self._trajectories = items

    def __len__(self) -> int:
        return len(self._trajectories)

    def __repr__(self) -> str:
        return f"TrajectoryEnsemble({len(self)} replicas)"

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        """Return the replicas in order."""
        return self._trajectories

    @property
    def topology(self) -> ChainTopology:
        """Return the topology shared by every replica."""
        return self._trajectories[0].topology

    def evaluate(
        self,
        observable: Observable,
        *,
        times: Sequence[float] | None = None,
        execution: ExecutionPolicy | None = None,
        retain_samples: bool = False,
        random_seed: object = None,
        **parameters: Any,
    ) -> TimeResolvedResult:
        """Evaluate a frame observable at each time and average across replicas.

        ``times=None`` uses every frame and requires all replicas to share one
        step grid; explicit ``times`` must exist in every replica. The
        observable receives a :class:`~polyequil.configuration.ChainConfiguration`.
        """
        indices, axis = self._frame_indices(times)
        grid = [
            [
                FrameRef(traj.path, int(column[r]))
                for r, traj in enumerate(self._trajectories)
            ]
            for column in indices
        ]
        return self._run(
            "frame",
            axis,
            grid,
            observable,
            execution,
            retain_samples,
            random_seed,
            parameters,
        )

    def evaluate_windows(
        self,
        observable: Observable,
        *,
        windows: Sequence[tuple[float, float]],
        execution: ExecutionPolicy | None = None,
        retain_samples: bool = False,
        random_seed: object = None,
        **parameters: Any,
    ) -> TimeResolvedResult:
        """Evaluate a window observable on each ``[t_start, t_stop]`` window.

        The observable receives a :class:`~polyequil.trajectory.WindowData`
        holding every frame of the window from one replica.
        """
        axis = np.array(windows, dtype=np.float64).reshape(-1, 2)
        if axis.shape[0] == 0 or np.any(axis[:, 1] < axis[:, 0]):
            raise ValidationError("windows must be non-empty (t_start <= t_stop) pairs")
        grid = [
            [traj.window(float(t0), float(t1)) for traj in self._trajectories]
            for t0, t1 in axis
        ]
        return self._run(
            "window",
            axis,
            grid,
            observable,
            execution,
            retain_samples,
            random_seed,
            parameters,
        )

    def _frame_indices(
        self, times: Sequence[float] | None
    ) -> tuple[list[NDArray[np.int64]], FloatArray]:
        """Return per-slice replica frame indices and the slice times."""
        first = self._trajectories[0]
        if times is None:
            for traj in self._trajectories[1:]:
                if not np.array_equal(traj.steps, first.steps):
                    raise ValidationError(
                        f"{traj.path.name} has a different step grid from "
                        f"{first.path.name}; pass explicit times="
                    )
            count = len(first)
            columns = [np.full(len(self), i, dtype=np.int64) for i in range(count)]
            return columns, np.array(first.times, dtype=np.float64)
        axis = np.array(times, dtype=np.float64).reshape(-1)
        if axis.size == 0:
            raise ValidationError("times must not be empty")
        columns = [
            np.array([traj.index_of_time(float(t)) for traj in self._trajectories])
            for t in axis
        ]
        return columns, axis

    def _run(
        self,
        kind: str,
        axis: FloatArray,
        grid: list[list[Any]],
        observable: Observable,
        execution: ExecutionPolicy | None,
        retain_samples: bool,
        random_seed: object,
        parameters: dict[str, Any],
    ) -> TimeResolvedResult:
        """Validate, execute (slice-major, replica-minor), and reduce per slice."""
        policy = _validated_call(observable, execution, retain_samples, parameters)
        accepts_rng = _validate_observable(observable)
        if random_seed is not None and not accepts_rng:
            raise ValidationError("random_seed given but the observable takes no rng")
        root, randomness = root_stream(random_seed) if accepts_rng else (None, None)
        m = len(self)
        payloads: list[Payload] = [
            (handle, observable, parameters, 1, root, (r, t))
            for t, row in enumerate(grid)
            for r, handle in enumerate(row)
        ]
        worker = _evaluate_frame if kind == "frame" else _evaluate_window
        label = self._labeller(kind, axis)
        stores = [_new_accumulator(retain_samples) for _ in grid]
        reference = _reduce(
            execute(policy, worker, payloads, label=label), stores, m, label
        )
        metadata = self._metadata(
            kind, observable, parameters, reference, policy, randomness
        )
        return _assemble(kind, axis, stores, m, reference, metadata, retain_samples)

    def _labeller(self, kind: str, axis: FloatArray) -> Callable[[int], str]:
        """Return a function naming flat task ``i`` for error messages."""
        names = [traj.path.name for traj in self._trajectories]
        m = len(names)

        def label(index: int) -> str:
            t, r = divmod(index, m)
            where = (
                f"t = {axis[t]:g}" if kind == "frame" else f"window {axis[t].tolist()}"
            )
            return f"replica {r} ({names[r]}) at {where}"

        return label

    def _metadata(
        self,
        kind: str,
        observable: Observable,
        parameters: Mapping[str, Any],
        reference: ObservableResult,
        policy: ExecutionPolicy,
        randomness: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Return provenance for the result."""
        return {
            "kind": kind,
            "observable": _observable_name(observable),
            "parameters": freeze_mapping(
                parameters, role="parameters", allow_opaque=True
            ),
            "convention": reference.convention,
            "replica_files": tuple(str(t.path) for t in self._trajectories),
            "root_seeds": tuple(
                t.attrs.get("root_seed", -1) for t in self._trajectories
            ),
            "weighting": "equal across replicas",
            "uncertainty": {
                "standard_deviation": "sample_ddof_1 across replicas",
                "standard_error": "standard_deviation/sqrt(replica_count)",
                "assumption": "independent replicas",
            },
            "execution": policy.describe(),
            "randomness": randomness,
        }


# ============================================================
# WORKERS (module level so they pickle)
# ============================================================


def _evaluate_frame(payload: Payload) -> ObservableResult:
    """Load one frame and evaluate the observable on it."""
    ref: FrameRef = payload[0]
    return _call(ref.load(), payload)


def _evaluate_window(payload: Payload) -> ObservableResult:
    """Load one window and evaluate the observable on it."""
    window: TrajectoryWindow = payload[0]
    return _call(window.load(), payload)


def _call(item: object, payload: Payload) -> ObservableResult:
    """Call the observable with the reserved keywords and check its return."""
    _, observable, parameters, threads, root, key = payload
    if root is None:
        result = observable(item, threads=threads, **parameters)
    else:
        result = observable(
            item, threads=threads, rng=child_generator(root, key), **parameters
        )
    if not isinstance(result, ObservableResult):
        raise ValidationError("an observable must return an ObservableResult")
    return result


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _check_independent(items: tuple[Trajectory, ...]) -> None:
    """Refuse repeated files, differing topologies, and repeated root seeds."""
    paths = [t.path for t in items]
    if len(set(paths)) != len(paths):
        raise ValidationError("the same trajectory file appears more than once")
    for traj in items[1:]:
        if traj.topology != items[0].topology:
            raise ValidationError(f"{traj.path.name} has a different topology")
    seeds = [t.attrs["root_seed"] for t in items if "root_seed" in t.attrs]
    if len(set(seeds)) != len(seeds):
        raise ValidationError(
            "two replicas share a root_seed; they are not independent"
        )


def _validated_call(
    observable: Observable,
    execution: ExecutionPolicy | None,
    retain_samples: bool,
    parameters: Mapping[str, Any],
) -> ExecutionPolicy:
    """Check every caller argument before any task runs; return the policy."""
    if execution is not None and not isinstance(execution, ExecutionPolicy):
        raise ValidationError("execution must be an ExecutionPolicy or None")
    if not isinstance(retain_samples, bool):
        raise ValidationError("retain_samples must be a bool")
    reserved = [name for name in RESERVED_PARAMETERS if name in parameters]
    if reserved:
        raise ValidationError(
            f"{reserved} are injected by the ensemble; pass random_seed= for rng"
        )
    for value, where in ((parameters, "parameters"), (observable, "observable")):
        found = find_random_state(value, where)
        if found is not None:
            raise ValidationError(
                f"{found} contains random state; stochastic observables must "
                "declare rng and receive streams through random_seed="
            )
    policy = execution or ExecutionPolicy()
    if policy.mode != "serial":
        try:
            pickle.dumps((observable, dict(parameters)))
        except Exception as exc:
            raise ValidationError(
                "parallel execution needs a picklable, module-level observable and "
                f"parameters ({exc})"
            ) from exc
    return policy


def _validate_observable(observable: Observable) -> bool:
    """Require a keyword-bindable ``threads``; return whether ``rng`` is declared."""
    if not callable(observable):
        raise ValidationError("an observable must be callable")
    try:
        signature = inspect.signature(observable)
    except (TypeError, ValueError) as exc:
        raise ValidationError("an observable needs an inspectable signature") from exc
    params = signature.parameters
    threads = params.get("threads")
    var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if (threads is None and not var_keyword) or (
        threads is not None and threads.kind not in _BY_KEYWORD
    ):
        raise ValidationError(
            "an observable must accept a keyword threads parameter, as in "
            "observable(item, *, threads=1, **parameters)"
        )
    rng = params.get("rng")
    if rng is not None and rng.kind not in _BY_KEYWORD:
        raise ValidationError("an observable rng parameter must be keyword-bindable")
    return rng is not None


def _observable_name(observable: Observable) -> str:
    """Return a module-qualified name for provenance."""
    qualname = getattr(observable, "__qualname__", None)
    if qualname is None:
        return repr(observable)
    module = getattr(observable, "__module__", None)
    return f"{module}.{qualname}" if module else str(qualname)


def _new_accumulator(retain_samples: bool) -> Accumulator:
    """Return an empty accumulator, retaining samples on request."""
    inner = WelfordAccumulator()
    return SampleStore(inner) if retain_samples else inner


def _reduce(
    stream: Generator[tuple[int, Any], None, None],
    stores: list[Accumulator],
    m: int,
    label: Callable[[int], str],
) -> ObservableResult:
    """Fold the ordered task stream into per-slice accumulators.

    Returns the first task's result, the compatibility reference.
    """
    reference: ObservableResult | None = None
    with contextlib.closing(stream) as results:
        for index, result in results:
            if reference is None:
                reference = result
            reason = results_compatible(reference, result)
            if reason is not None:
                raise EnsembleError(
                    f"{label(index)} is incompatible with {label(0)}: {reason}",
                    task_indices=(index,),
                )
            stores[index // m].update(result)
    if reference is None:
        raise ValidationError("an evaluation needs at least one task")
    return reference


def _assemble(
    kind: str,
    axis: FloatArray,
    stores: list[Accumulator],
    m: int,
    reference: ObservableResult,
    metadata: dict[str, Any],
    retain_samples: bool,
) -> TimeResolvedResult:
    """Stack per-slice statistics into a :class:`TimeResolvedResult`."""
    outcomes = [store.result() for store in stores]
    if any(count != m for _, _, count in outcomes):
        raise EnsembleError("a time slice did not receive every replica")
    mean = np.stack([mean for mean, _, _ in outcomes])
    deviation = error = None
    if m > 1:
        variances = [variance for _, variance, _ in outcomes]
        deviation = np.sqrt(np.stack([v for v in variances if v is not None]))
        error = deviation / np.sqrt(float(m))
    samples = None
    if retain_samples:
        samples = np.stack([s.samples() for s in stores if isinstance(s, SampleStore)])
    return TimeResolvedResult(
        kind, axis, mean, deviation, error, m, reference.coordinates, metadata, samples
    )


def _validate_uncertainty(result: TimeResolvedResult, shape: tuple[int, ...]) -> None:
    """Enforce the single-replica / multi-replica uncertainty contract."""
    deviation, error = result.standard_deviation, result.standard_error
    if result.replica_count == 1:
        if deviation is not None or error is not None:
            raise ValidationError("a single-replica result has no uncertainty")
        return
    if deviation is None or error is None:
        raise ValidationError("a multi-replica result needs deviation and error")
    frozen = []
    for name, value in (("standard_deviation", deviation), ("standard_error", error)):
        array = np.array(value, dtype=np.float64)
        if array.shape != shape or np.any(array < 0.0):
            raise ValidationError(f"{name} must be non-negative with shape {shape}")
        array.setflags(write=False)
        frozen.append(array)
    expected = frozen[0] / np.sqrt(float(result.replica_count))
    if not np.allclose(
        frozen[1], expected, rtol=32 * np.finfo(float).eps, atol=0.0, equal_nan=True
    ):
        raise ValidationError(
            "standard_error must equal deviation / sqrt(replica_count)"
        )
    object.__setattr__(result, "standard_deviation", frozen[0])
    object.__setattr__(result, "standard_error", frozen[1])
