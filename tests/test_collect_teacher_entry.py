"""The real collection entry initializes PackedEnvironment before reading its device."""
from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.course import factory as course_factory
from supermimic_rl.adapters.packed import PackedEnvironment
from supermimic_rl.world_model_v2 import teacher
from supermimic_rl.world_model_v2.replay import CompactReplay


@pytest.mark.parametrize('failure', [None, 'teacher', 'collect'])
@pytest.mark.parametrize('record_recovery', [False, True])
def test_entry_sets_course_then_resets_then_loads_teacher_and_persists_errors_before_close(
        tmp_path, monkeypatch, failure, record_recovery):
    spec = importlib.util.spec_from_file_location('collect_teacher_entry',
        Path(__file__).resolve().parents[1] / 'scripts/collect_teacher_v2.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    output, events = tmp_path / 'collection', []
    configuration = tmp_path / 'factory.json'
    configuration.write_text(json.dumps({'record_substeps': True, 'seed': 7}))

    class Adapter:
        provenance, policy_dt_s = {}, .02

        def set_training_progress(self, **progress):
            assert progress == {'real_transitions': 52428800, 'actor_updates': 0}
            events.append('coordinate')

        def reset(self):
            assert events == ['coordinate']
            events.append('reset')
            return SimpleNamespace(state={'physical': torch.zeros(2, 1)}, context={},
                observation=torch.zeros(2, 35), goal=torch.zeros(2, 11),
                episode_id=torch.zeros(2, dtype=torch.int64), control_epoch=torch.zeros(2, dtype=torch.int64),
                policy_owned=torch.ones(2, dtype=torch.bool))

        def close(self):
            if failure is not None:
                text = (output / 'failure-traceback.txt').read_text()
                assert 'RuntimeError: collection entry ' + failure in text
            events.append('close')
            if failure is not None:
                raise SystemExit(0)  # Isaac may exit before the outer traceback printer.

    adapter = Adapter()
    if record_recovery:
        from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig
        from supermimic_rl.world_model_v2.recovery_task import RecoveryTaskConfig
        from supermimic_rl.world_model_v2.stair_source import StairScenario

        @dataclass(frozen=True)
        class TaskConfig:
            fixture: bool = True

        adapter.env = SimpleNamespace(_terrain_course=SimpleNamespace(
            tof_trigger=SimpleNamespace(generator=torch.Generator())))
        monkeypatch.setattr(StairScenario, 'capture', lambda native: SimpleNamespace(to_dict=lambda: {'fixture': True}))
        monkeypatch.setattr(JumpTaskConfig, 'capture', lambda native: TaskConfig())
        monkeypatch.setattr(RecoveryTaskConfig, 'capture', lambda native: TaskConfig())
    # The fixture intentionally uses the real wrapper: device is absent until
    # reset packs native tensors, not available at construction from an adapter.
    assert not hasattr(PackedEnvironment(adapter, source='{}'), 'device')
    monkeypatch.setattr(course_factory, 'create_course', lambda configuration: adapter)
    monkeypatch.setattr(entry, 'native_logs', lambda path: nullcontext())

    def load(path, digest, *, device, native_provenance):
        assert events == ['coordinate', 'reset']
        assert device == torch.device('cpu') and native_provenance == {}
        events.append('teacher')
        if failure == 'teacher':
            raise RuntimeError('collection entry teacher')
        return SimpleNamespace(source_metadata={'fixture': True})

    class Collector:
        def __init__(self, env, decision, **configuration):
            assert isinstance(env, PackedEnvironment) and env.device == torch.device('cpu')
            assert events == ['coordinate', 'reset', 'teacher']
            assert configuration['course_coordinate'] == 52428800
            for name in ('terrain', 'tasks', 'stairs', 'stair_source', 'jumps', 'recovery'):
                assert configuration['record_' + name] == record_recovery
            self.environment = env
            self.record_stair_source = self.record_jumps = self.record_recovery = record_recovery
            self.replay = CompactReplay({'frame': (35,)}, 2, 1, 'cpu')

        def collect(self, steps, **callbacks):
            events.append('collect')
            if failure == 'collect':
                raise RuntimeError('collection entry collect')
            boundary = {name: torch.ones(2, dtype=torch.bool) for name in
                        ('valid', 'policy_owned', 'end_policy_owned')}
            boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
            boundary.update({name: torch.zeros(2, dtype=torch.int64) for name in
                             ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
            self.replay.add({'frame': torch.zeros(2, 35)}, boundary)
            if record_recovery:
                callbacks['on_source_rng'](0, torch.Generator().get_state())
            return {'transitions': 2}

    monkeypatch.setattr(teacher, 'load_native_teacher', load)
    monkeypatch.setattr(entry, 'Collector', Collector)
    monkeypatch.setattr('sys.argv', ['collect_teacher_v2', '--factory-config', str(configuration),
        '--checkpoint', str(tmp_path / 'teacher.pt'), '--checkpoint-sha256', 'fixture',
        '--course-coordinate', '52428800', '--steps', '1', '--output', str(output),
        *(['--record-recovery'] if record_recovery else [])])
    if failure is None:
        entry.main()
        assert events == ['coordinate', 'reset', 'teacher', 'collect', 'close']
        assert (output / 'replay.pt').is_file()
        if record_recovery:
            payload = torch.load(output / 'replay.pt', weights_only=True)
            assert payload['recovery_task_config'] == payload['jump_task_config'] == {'fixture': True}
            assert json.loads((output / 'recovery-task-config.json').read_text()) == {'fixture': True}
    else:
        with pytest.raises(SystemExit):
            entry.main()
        assert events[-1] == 'close'
        assert not (output / 'replay.pt').exists()
