"""Native close must not turn failed V2 work into a successful process exit."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
from contextlib import nullcontext
import errno
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

root, directory, operation, failure, receipt_failure = sys.argv[1:]
directory = Path(directory)
output = directory / 'output'

def install(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module

class Native:
    device = 'cpu'
    schema = SimpleNamespace(fingerprint='fixture')

    def reset(self):
        if failure == 'reset':
            raise RuntimeError('native reset sentinel')

    def close(self):
        (directory / 'close-called').write_text('native exit-zero close')
        os._exit(0)

class Trainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(self, **kwargs):
        if failure == 'work':
            raise OSError(errno.ENOSPC, 'world checkpoint ENOSPC sentinel')
        return {'status': 'complete', 'cycles': 128}

def evaluate(*args, **kwargs):
    if failure == 'work':
        raise RuntimeError('native evaluation sentinel')
    return {'steps': 3}

install('supermimic_rl.adapters.packed', PackedEnvironment=lambda *a, **k: Native())
install('supermimic_rl.cli', native_logs=lambda *a: nullcontext(), _native_schema_source=lambda p: 'fixture')
install('supermimic_rl.course.factory', create_course=lambda config: SimpleNamespace(provenance={}, policy_dt_s=.02))
install('supermimic_rl.world_model_v2.online', OnlineConfig=SimpleNamespace(from_dict=lambda x: x), OnlineTrainer=Trainer)
install('supermimic_rl.world_model_v2.teacher', load_native_teacher=lambda *a, **k: SimpleNamespace(source_metadata={}),
        native_teacher_decision=lambda *a: None, file_sha256=lambda path: 'fixture')
install('supermimic_rl.world_model_v2.evaluation', evaluate_native=evaluate)
install('supermimic_rl.world_model_v2.student_transfer',
        load_student_actor=lambda *a: (None, {'schema_fingerprint': 'fixture'}))
runtime, factory, config = (directory / name for name in ('runtime.json', 'factory.json', 'config.json'))
runtime.write_text('{}')
factory.write_text(json.dumps({'runtime_json': str(runtime), 'asset_directory': 'fixture', 'identified_bundle': 'fixture'}))
config.write_text('{}')
if receipt_failure == 'yes':
    original_write = Path.write_text
    def write(self, text, *args, **kwargs):
        if self.name == 'failure-traceback.txt':
            raise OSError(errno.ENOSPC, 'receipt ENOSPC sentinel')
        return original_write(self, text, *args, **kwargs)
    Path.write_text = write
sys.argv = ['train_world_v2.py', operation, '--factory-config', str(factory), '--output', str(output)]
if operation == 'train':
    sys.argv += ['--config', str(config), '--teacher', 'fixture', '--teacher-sha256', 'fixture']
else:
    sys.argv += ['--checkpoint', 'fixture', '--steps', '3']
runpy.run_path(str(Path(root) / 'scripts/train_world_v2.py'), run_name='__main__')
'''


def child(tmp_path, operation, failure='none', receipt_failure=False):
    return subprocess.run([sys.executable, '-c', CHILD, str(ROOT), str(tmp_path), operation,
                           failure, 'yes' if receipt_failure else 'no'],
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('operation', ['train', 'evaluate'])
@pytest.mark.parametrize('failure', ['reset', 'work'])
def test_native_failure_exits_nonzero_before_native_exit_zero_close(tmp_path, operation, failure):
    result = child(tmp_path, operation, failure)
    expected = ('native reset sentinel' if failure == 'reset' else
                'world checkpoint ENOSPC sentinel' if operation == 'train' else 'native evaluation sentinel')
    assert result.returncode == 1, result.stderr
    assert expected in result.stderr
    assert expected in (tmp_path / 'output/failure-traceback.txt').read_text()
    assert not (tmp_path / 'close-called').exists()
    assert not (tmp_path / 'output/summary.json').exists()


def test_original_save_error_still_exits_nonzero_when_failure_receipt_disk_is_full(tmp_path):
    result = child(tmp_path, 'train', 'work', receipt_failure=True)
    assert result.returncode == 1, result.stderr
    assert 'world checkpoint ENOSPC sentinel' in result.stderr
    assert 'Could not write failure-traceback.txt' in result.stderr
    assert 'receipt ENOSPC sentinel' in result.stderr
    assert not (tmp_path / 'close-called').exists()
    assert not (tmp_path / 'output/summary.json').exists()


@pytest.mark.parametrize('operation', ['train', 'evaluate'])
def test_success_persists_summary_then_keeps_native_exit_zero_close(tmp_path, operation):
    result = child(tmp_path, operation)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'close-called').is_file()
    report = json.loads((tmp_path / 'output/summary.json').read_text())
    assert report['cycles' if operation == 'train' else 'steps'] == (128 if operation == 'train' else 3)
    assert not (tmp_path / 'output/failure-traceback.txt').exists()
