"""
CPU-budgeted execution of independent tasks, yielded in index order.

Rewritten from heteromorph's ``execution.py`` with the deferred ``inner`` and
``hybrid`` modes and the chunked reduction removed, and threadpoolctl made a
required dependency. This module decides only how many processes and threads
may run; it holds no statistics.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import multiprocessing
import os
import sys
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any

# ----- third party -----
from threadpoolctl import threadpool_limits

# ----- local -----
from polyequil.errors import EnsembleError, ValidationError

# ============================================================
# CONSTANTS
# ============================================================

MODES: tuple[str, ...] = ("serial", "outer")
START_METHODS: tuple[str, ...] = ("forkserver", "spawn")
SUBMISSION_WINDOW_PER_WORKER: int = 2
THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# ============================================================
# TYPES
# ============================================================


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """An explicit CPU budget for evaluating tasks.

    ``serial`` (the default and the reference) runs in-process; ``outer`` runs
    one single-threaded worker process per CPU, up to ``ensemble_workers``.
    Every field is resolved to a concrete value at construction, and
    ``ensemble_workers * threads_per_worker <= total_cpus`` is enforced. The
    CPU budget comes from the argument, else ``SLURM_CPUS_PER_TASK``, else the
    process affinity mask, else ``os.cpu_count()``.
    """

    mode: str = "serial"
    total_cpus: int | None = None
    ensemble_workers: int | None = None
    threads_per_worker: int | None = None
    start_method: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValidationError(f"mode must be one of {MODES}, not {self.mode!r}")
        total = _resolve_total_cpus(self.total_cpus)
        if self.threads_per_worker not in (None, 1):
            raise ValidationError(
                f"{self.mode} mode runs one thread per worker; more threads per "
                "task are not implemented"
            )
        if self.mode == "serial":
            if self.ensemble_workers not in (None, 1):
                raise ValidationError("serial mode uses one worker; use mode='outer'")
            workers = 1
        else:
            workers = total if self.ensemble_workers is None else self.ensemble_workers
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValidationError("ensemble_workers must be a positive int")
        if workers > total:
            raise ValidationError(
                f"ensemble_workers ({workers}) exceeds the CPU budget ({total})"
            )
        method = self.start_method
        if method is None:
            method = "forkserver" if sys.platform.startswith("linux") else "spawn"
        if method not in START_METHODS:
            raise ValidationError(
                f"start_method must be one of {START_METHODS}, not {method!r}; "
                "plain fork interacts badly with initialized native thread pools"
            )
        object.__setattr__(self, "total_cpus", total)
        object.__setattr__(self, "ensemble_workers", workers)
        object.__setattr__(self, "threads_per_worker", 1)
        object.__setattr__(self, "start_method", method)

    def worker_count(self, task_count: int) -> int:
        """Return the process count to use, never more than there is work for."""
        assert self.ensemble_workers is not None, "resolved at construction"
        return min(self.ensemble_workers, max(1, task_count))

    def describe(self) -> dict[str, object]:
        """Return the resolved budget for provenance metadata."""
        return {
            "mode": self.mode,
            "ensemble_workers": self.ensemble_workers,
            "threads_per_worker": self.threads_per_worker,
            "total_cpus": self.total_cpus,
            "start_method": self.start_method,
        }


# ============================================================
# PUBLIC API
# ============================================================


def execute(
    policy: ExecutionPolicy,
    worker_function: Callable[[Any], Any],
    payloads: Iterable[Any],
    *,
    label: Callable[[int], str] = lambda index: f"task {index}",
) -> Generator[tuple[int, Any], None, None]:
    """Yield ``(index, result)`` pairs in ascending index order.

    Index-ordered consumption fixes the reduction sequence, so every mode
    aggregates bit-identically. In ``outer`` mode the generator owns a process
    pool while suspended; a caller that may abandon it early must close it
    (``contextlib.closing``).

    Parameters
    ----------
    policy : ExecutionPolicy
        Resolved CPU budget.
    worker_function : Callable[[Any], Any]
        Module-level (picklable) function applied to each payload.
    payloads : Iterable[Any]
        One payload per task.
    label : Callable[[int], str]
        Human-readable task name for error messages.

    Raises
    ------
    EnsembleError
        When a task raises (definitive index) or a worker dies (candidates).
    """
    if policy.mode == "serial":
        yield from _execute_serially(worker_function, payloads, label)
    else:
        yield from _execute_pooled(policy, worker_function, list(payloads), label)


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _resolve_total_cpus(requested: int | None) -> int:
    """Resolve the CPU budget, preferring a scheduler allocation to the machine."""
    if requested is not None:
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise ValidationError("total_cpus must be an int")
        if requested < 1:
            raise ValidationError("total_cpus must be at least one")
        return requested
    allocation = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if allocation.isdigit() and int(allocation) >= 1:
        return int(allocation)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
        except OSError:
            count = 0
        if count >= 1:
            return count
    return int(os.cpu_count() or 1)


def _task_failure(index: int, exc: BaseException, label: str) -> EnsembleError:
    """Build the error for one task whose observable raised (both modes)."""
    return EnsembleError(f"{label} failed: {exc}", task_indices=(index,))


def _execute_serially(
    worker_function: Callable[[Any], Any],
    payloads: Iterable[Any],
    label: Callable[[int], str],
) -> Generator[tuple[int, Any], None, None]:
    """Run tasks in-process, one at a time."""
    for index, payload in enumerate(payloads):
        try:
            result = worker_function(payload)
        except Exception as exc:
            raise _task_failure(index, exc, label(index)) from exc
        yield index, result


def _limit_worker_threads(threads: int) -> None:
    """Pool initializer: limit native thread pools already loaded in a child."""
    threadpool_limits(limits=int(threads))


class _ThreadEnvironment:
    """Temporarily publish a per-worker thread limit for children to inherit.

    A fallback only: under forkserver a worker inherits the forkserver's
    environment snapshot, so the threadpoolctl initializer is authoritative.
    """

    def __init__(self, threads: int) -> None:
        self._threads = threads
        self._previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        self._previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        for name in THREAD_VARIABLES:
            os.environ[name] = str(self._threads)

    def __exit__(self, *exc_info: object) -> None:
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _pool_failure(
    index: int, pending: dict[int, Future[Any]], label: Callable[[int], str]
) -> EnsembleError:
    """Build the error for a worker that died; in-flight tasks are candidates."""
    candidates = tuple(sorted({index, *pending}))
    names = ", ".join(label(i) for i in candidates)
    detail = (
        "a worker process terminated before returning; the candidate in-flight "
        f"tasks were {candidates} ({names}). Which one caused the termination "
        "cannot be determined from the parent process."
    )
    if index == 0:
        detail += (
            " No task completed first, which usually means the pool failed to "
            "start: check that the calling script guards its entry point with "
            'if __name__ == "__main__":.'
        )
    return EnsembleError(detail, task_indices=candidates)


def _execute_pooled(
    policy: ExecutionPolicy,
    worker_function: Callable[[Any], Any],
    payloads: Sequence[Any],
    label: Callable[[int], str],
) -> Generator[tuple[int, Any], None, None]:
    """Run tasks in a process pool with a bounded sliding submission window."""
    total = len(payloads)
    workers = policy.worker_count(total)
    window = SUBMISSION_WINDOW_PER_WORKER * workers
    threads = policy.threads_per_worker or 1
    context = multiprocessing.get_context(policy.start_method)
    with (
        _ThreadEnvironment(threads),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_limit_worker_threads,
            initargs=(threads,),
        ) as pool,
    ):
        pending: dict[int, Future[Any]] = {}
        submitted = 0
        try:
            for index in range(total):
                # Never Executor.map (it submits everything eagerly). submit()
                # sits inside the guard because a broken pool raises there too.
                try:
                    while submitted < total and len(pending) < window:
                        pending[submitted] = pool.submit(
                            worker_function, payloads[submitted]
                        )
                        submitted += 1
                    result = pending.pop(index).result()
                except BrokenProcessPool as exc:
                    raise _pool_failure(index, pending, label) from exc
                except Exception as exc:
                    raise _task_failure(index, exc, label(index)) from exc
                yield index, result
        finally:
            # The pool's own shutdown never cancels queued futures.
            for future in pending.values():
                future.cancel()
