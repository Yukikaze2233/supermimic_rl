"""Resuming a process must not replay historical native output as new physics."""
import subprocess
import sys


CHILD = r'''
import os
from pathlib import Path
import sys

from supermimic_rl.cli import native_logs
from transformer_rl.formal.simulation.health import NativePhysicsHealth

directory, current_error = Path(sys.argv[1]), sys.argv[2] == 'error'
with native_logs(directory):
    health = NativePhysicsHealth.for_native_process(directory)
    try:
        os.write(1, b'current worker output\n')
        health.check()
        if current_error:
            os.write(2, b'[omni.physx.plugin] PhysX error: current sentinel\n')
            health.check()
    finally:
        health.close()
print('restored output descriptor')
'''


def run_worker(directory, error=False):
    return subprocess.run([sys.executable, '-c', CHILD, str(directory),
                           'error' if error else 'ok'],
                          capture_output=True, text=True, timeout=30)


def test_large_prior_worker_logs_are_preserved_without_reentering_health_reader(tmp_path):
    # Reproduce the formal WM restart failure: old output alone exceeds the
    # native reader's 8 MiB limit before the new process even starts physics.
    previous_stdout = b'old metric\n' * 850_000
    previous_stderr = b'[omni.physx.plugin] PhysX error: historical sentinel\n'
    (tmp_path / 'native.stdout.log').write_bytes(previous_stdout)
    (tmp_path / 'native.stderr.log').write_bytes(previous_stderr)
    result = run_worker(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'restored output descriptor' in result.stdout
    for name, content in (('stdout', previous_stdout), ('stderr', previous_stderr)):
        archives = list(tmp_path.glob(f'native.{name}.previous-*.log'))
        assert len(archives) == 1 and archives[0].read_bytes() == content
    assert (tmp_path / 'native.stdout.log').read_bytes() == b'current worker output\n'
    assert b'historical sentinel' not in (tmp_path / 'native.stderr.log').read_bytes()


def test_current_worker_physics_errors_still_raise_and_output_is_restored(tmp_path):
    result = run_worker(tmp_path, error=True)
    assert result.returncode != 0
    assert 'native physics reported an invalid interval' in result.stderr
    assert 'current sentinel' in result.stderr
    assert b'current sentinel' in (tmp_path / 'native.stderr.log').read_bytes()
    assert not list(tmp_path.glob('*.previous-*.log'))
