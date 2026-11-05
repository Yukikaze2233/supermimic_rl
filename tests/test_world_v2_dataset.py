"""Native dataset serialization preserves learning targets and source identity."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.world_model_v2.dataset import save_native_dataset
from supermimic_rl.world_model_v2.ingest import load_recorded_replay
from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT, RecordSpec
from supermimic_rl.world_model_v2.recovery_task import RecoveryTaskConfig
from supermimic_rl.world_model_v2.replay import CompactReplay
from supermimic_rl.world_model_v2.stair_source import StairScenario
from support.world_v2.recovery_task import native_recovery
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import native_stairs


def recorded_collector(full=True):
    native = native_stairs(2)
    recovery = native_recovery(2)
    native.physics_dt = recovery.physics_dt
    native._terrain_course.tasks = recovery._terrain_course.tasks
    native._terrain_course.recovery = recovery._terrain_course.recovery
    _, native._terrain_course.tof_trigger = source_fixture(2)
    schema = StateSchema((Field('state', (1,)),), (Field('context', (1,)),),
                         observation_dim=45, goal_dim=16,
                         substep_fields=(Field('tick', (1,)),), substeps_per_step=4)
    fields = RecordSpec(16, 65).fields(
        record_terrain=full, record_tasks=full, record_stairs=full,
        record_stair_source=full, record_jumps=full, record_recovery=full)
    replay = CompactReplay(fields, 2, 4, 'cpu')
    rng, source_rng = native._terrain_course.tof_trigger.generator, []
    for step in range(4):
        record = {key: torch.zeros_like(replay.data[key][0]) for key in fields}
        record['physical'][:, 6] = record['end_physical'][:, 6] = 1.
        record['frame'].fill_(step)
        record['action'].fill_(step * .1)
        record['reward'].fill_(-step)  # Failed attempts are retained too.
        record['ticks'].fill_(step + .25)
        if full:
            record['jump_ids'].fill_(2**54 + step)
            record['stair_clock'].fill_(1_000_000.01 + step * .02)
            record['recovery_flags'][:, 0] = True
        boundary = {name: torch.ones(2, dtype=torch.bool) for name in
                    ('valid', 'policy_owned', 'end_policy_owned')}
        boundary.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
        boundary.update({name: torch.zeros(2, dtype=torch.int64) for name in
                         ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
        boundary['valid'][1] = step != 1
        boundary['terminated'][0] = step == 2
        boundary['policy_owned'][1] = step >= 2
        boundary['end_policy_owned'][1] = step >= 2
        boundary['start_episode_id'][0] = boundary['end_episode_id'][0] = int(step == 3)
        source_rng.append((step, rng.get_state().clone()))
        torch.rand(1, generator=rng)
        replay.add(record, boundary)
    return SimpleNamespace(
        environment=SimpleNamespace(num_envs=2, adapter=SimpleNamespace(env=native), schema=schema),
        replay=replay, record_stair_source=full, record_jumps=full, record_recovery=full), source_rng


@pytest.mark.parametrize('full', [False, True])
def test_native_dataset_roundtrip_preserves_every_field_boundary_and_source(tmp_path, full):
    collector, source_rng = recorded_collector(full)
    output = tmp_path / 'replay.pt'
    collection = {'terminations': 1, 'invalid_transitions': 1, 'transitions': 8}
    source = {'checkpoint_sha256': 'a' * 64, 'action_sampling': {'mode': 'ppo-gaussian', 'seed': 7}}
    generator = collector.environment.adapter.env._terrain_course.tof_trigger.generator
    rng_before = generator.get_state().clone()
    report = save_native_dataset(collector, output, collection, source_rng=source_rng, source=source)
    replay, schema = load_recorded_replay(output, record_terrain=full)
    payload = torch.load(output, weights_only=True)
    assert schema == collector.environment.schema
    assert payload['model_contract'] == MODEL_CONTRACT
    assert payload['collection'] == collection and payload['source'] == source
    assert report['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report['transitions'] == 8 and report['vector_steps'] == 4
    for key, value in collector.replay.state_dict()['storage'].items():
        torch.testing.assert_close(replay.data[key], value, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), rng_before, rtol=0, atol=0)
    if full:
        assert replay.data['jump_ids'].dtype == torch.int64
        assert int(replay.data['jump_ids'][3, 0, 0]) == 2**54 + 3
        assert replay.data['stair_clock'].dtype == torch.float64
        assert float(replay.data['stair_clock'][2, 0, 0]) == 1_000_000.05
        assert replay.data['recovery_flags'].dtype == torch.bool
        torch.testing.assert_close(payload['source_rng']['sequence'], torch.arange(4))
        torch.testing.assert_close(payload['source_rng']['state'], torch.stack([x[1] for x in source_rng]))
        native = collector.environment.adapter.env
        expected = {'stair-source-config.json': ('stair_source_config', StairScenario.capture(native).to_dict()),
                    'jump-task-config.json': ('jump_task_config', asdict(JumpTaskConfig.capture(native))),
                    'recovery-task-config.json': ('recovery_task_config', asdict(RecoveryTaskConfig.capture(native)))}
        for filename, (key, config) in expected.items():
            assert payload[key] == config
            assert json.loads((tmp_path / filename).read_text()) == json.loads(json.dumps(config))
    else:
        assert 'source_rng' not in payload and 'jump_task_config' not in payload


@pytest.mark.parametrize('rng_steps', [(0, 1, 3), (0, 2, 1, 3)])
def test_native_dataset_rejects_missing_or_misaligned_source_randomness(tmp_path, rng_steps):
    collector, source_rng = recorded_collector()
    output = tmp_path / 'replay.pt'
    with pytest.raises(ValueError, match='every recorded vector step'):
        save_native_dataset(collector, output, {}, source_rng=[source_rng[i] for i in rng_steps])
    assert not output.exists()


def test_native_dataset_does_not_overwrite_existing_data(tmp_path):
    collector, source_rng = recorded_collector()
    output = tmp_path / 'replay.pt'
    output.write_bytes(b'previous dataset')
    with pytest.raises(FileExistsError):
        save_native_dataset(collector, output, {}, source_rng=source_rng)
    assert output.read_bytes() == b'previous dataset'


def test_recording_is_passive_and_captures_evaluation_config_before_restore(tmp_path, monkeypatch):
    from supermimic_rl.world_model_v2 import collection, evaluation
    from support.world_v2.collection import Environment

    fixture, _ = recorded_collector()
    prototype = fixture.replay.data
    captures = ((collection.StairTaskState, 'stair_'), (collection.StairSourceState, 'tof_'),
                (collection.JumpTaskState, 'jump_'), (collection.RecoveryTaskState, 'recovery_'))
    for state_type, prefix in captures:
        values = {name: value[0].clone() for name, value in prototype.items()
                  if name.startswith(prefix) and not name.endswith('_valid')}
        monkeypatch.setattr(state_type, 'capture', lambda native, values=values:
                            SimpleNamespace(record=lambda: {k: v.clone() for k, v in values.items()}))
    monkeypatch.setattr(collection.MotionTask, 'capture', lambda native: prototype['motion_task'][0].clone())
    monkeypatch.setattr(evaluation, 'native_tracking_sample', lambda env:
                        SimpleNamespace(actual=torch.zeros(2, 3), target=torch.zeros(2, 3)))

    class Layout:
        def __init__(self, schema, *, record_terrain=False):
            self.fields = RecordSpec(16, 65).fields(record_terrain=record_terrain)

        def physical(self, state):
            return SimpleNamespace(gravity_body=torch.tensor([[0., 0., -1.]]).repeat(2, 1))

        def encode(self, transition):
            record = {name: prototype[name][0].clone() for name in self.fields}
            record.update(frame=transition.start.actor_observation.clone(), action=transition.action.clone(),
                          reward=transition.reward[:, None])
            return record

    monkeypatch.setattr(collection, 'SnapshotLayout', Layout)
    seen_collectors = []

    def collector(*args, **kwargs):
        result = collection.Collector(*args, **kwargs)
        seen_collectors.append(result)
        return result

    monkeypatch.setattr(evaluation, 'Collector', collector)

    def run(recording):
        source, _ = recorded_collector()
        env = Environment()
        env.schema, env.policy_dt_s = source.environment.schema, .02
        env.adapter = owner = source.environment.adapter
        owner._evaluation_configuration = {'fixed': True}
        env.current.state = torch.zeros(2, 1)
        owner.current = env.current
        env._pack = lambda snapshot: snapshot
        trigger = owner.env._terrain_course.tof_trigger
        owner.evaluation_begin = lambda schema: setattr(trigger, 'min_height_m', .08)
        owner.evaluation_end = lambda: setattr(trigger, 'min_height_m', .04)
        owner.evaluation_step = lambda *args: None
        owner.evaluation_summary = lambda: {'fixed': True}
        original_step = env.step
        actions, draws, histories = [], [], []

        def step(action):
            actions.append(action.clone())
            draws.append(torch.rand(2, generator=trigger.generator))
            transition = original_step(action)
            transition.end.state = env.current.state = torch.zeros(2, 1)
            return transition

        env.step = step

        class Actor:
            def eval(self):
                return self

            def __call__(self, history):
                histories.append(history.clone())
                return history[:, -1, :6] * .1

            def estimate(self, history):
                return torch.zeros(2, 5)

        report = evaluation.evaluate_native(env, Actor(), 4,
            record_dataset=tmp_path / 'recorded.pt' if recording else None)
        assert trigger.min_height_m == .04 and env.progress == []
        return report, torch.stack(actions), torch.stack(draws), torch.stack(histories), trigger.generator.get_state()

    plain, recorded = run(False), run(True)
    for before, after in zip(plain[1:], recorded[1:]):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert {k: v for k, v in plain[0].items() if k != 'elapsed_s'} == {
        k: v for k, v in recorded[0].items() if k not in ('elapsed_s', 'dataset')}
    assert seen_collectors[0].capacity == 1 and len(seen_collectors[0].replay) == 2
    assert seen_collectors[1].capacity == 4 and len(seen_collectors[1].replay) == 8
    saved = torch.load(tmp_path / 'recorded.pt', weights_only=True)
    assert saved['stair_source_config']['min_height_m'] == .08
    torch.testing.assert_close(saved['replay']['storage']['action'], recorded[1], rtol=0, atol=0)
    for state, draw in zip(saved['source_rng']['state'], recorded[2]):
        # CPU state readers need owning storage, not an offset row of the saved stack.
        rng = torch.Generator().set_state(state.clone())
        torch.testing.assert_close(torch.rand(2, generator=rng), draw, rtol=0, atol=0)
