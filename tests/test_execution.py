import contextlib

import pytest

from polyequil.errors import EnsembleError, ValidationError
from polyequil.execution import ExecutionPolicy, execute
from tests import observable_fixtures as fx


def _run(policy, worker, payloads):
    with contextlib.closing(execute(policy, worker, payloads)) as stream:
        return list(stream)


def test_policies_resolve_every_field():
    serial = ExecutionPolicy(total_cpus=4)
    assert (serial.ensemble_workers, serial.threads_per_worker) == (1, 1)
    outer = ExecutionPolicy("outer", total_cpus=4)
    assert outer.ensemble_workers == 4
    assert outer.start_method in ("forkserver", "spawn")
    assert outer.worker_count(2) == 2


def test_scheduler_allocation_outranks_the_machine(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert ExecutionPolicy("outer").total_cpus == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "inner"},
        {"mode": "outer", "total_cpus": 2, "ensemble_workers": 3},
        {"mode": "serial", "ensemble_workers": 2},
        {"threads_per_worker": 2},
        {"start_method": "fork"},
        {"total_cpus": 0},
    ],
)
def test_invalid_policies_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        ExecutionPolicy(**kwargs)


def test_serial_and_outer_agree_and_keep_index_order():
    payloads = [float(i) for i in range(50)]
    serial = _run(ExecutionPolicy(), fx.square, payloads)
    outer = _run(ExecutionPolicy("outer", total_cpus=2), fx.square, payloads)
    assert serial == outer == [(i, float(i) ** 2) for i in range(50)]


def test_failures_are_reported_identically_in_both_modes():
    messages = []
    for policy in (ExecutionPolicy(), ExecutionPolicy("outer", total_cpus=2)):
        with pytest.raises(EnsembleError) as info:
            _run(policy, fx.raise_on_three, list(range(6)))
        assert info.value.task_indices == (3,)
        messages.append(str(info.value))
    assert messages[0] == messages[1] == "task 3 failed: three is not allowed"


def test_worker_death_names_candidate_tasks():
    with pytest.raises(EnsembleError, match="terminated") as info:
        _run(ExecutionPolicy("outer", total_cpus=2), fx.exit_on_four, list(range(8)))
    assert 4 in info.value.task_indices
