"""Platform evidence preserves typed start history and actual contact validity."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.adapters.substeps import substep_fields
from supermimic_rl.contracts import Field, StateSchema
from supermimic_rl.course.jump_physics import PlatformLandingEvidence, PlatformLandingPhase, PlatformLandingState
from supermimic_rl.course.native_platform_landing import PlatformLandingInterval
from supermimic_rl.course.platform_intent import PlatformIntentPhase, PlatformIntentState
from supermimic_rl.world_model_v2.collection import Collector
from supermimic_rl.world_model_v2.layout import SnapshotLayout
from supermimic_rl.world_model_v2.physical import VELOCITY_JOINTS
from supermimic_rl.world_model_v2.platform_records import (
    PLATFORM_RECORD_DTYPES, PLATFORM_RECORD_FIELDS, PLATFORM_RECORD_VERSION, PlatformRecordState,
    decode_platform_interval, encode_platform_interval,
)
from supermimic_rl.world_model_v2.records import CONTACT_BODIES, PASSIVE_KNEE_JOINTS, RecordSpec, TICK_LABEL_DIM
from supermimic_rl.world_model_v2.replay import CompactReplay
from support.world_v2.collection import Environment


OPTIONS = dict(record_terrain=True, record_tasks=True, record_stairs=True,
               record_stair_source=True, record_jumps=True)


def platform_state():
    intent = PlatformIntentState.inactive(2)
    intent = replace(intent, target_index=torch.tensor([-1, 1]),
        phase=torch.tensor([int(PlatformIntentPhase.INACTIVE), int(PlatformIntentPhase.PHYSICAL_TAIL)]),
        bound_request_id=torch.tensor([-1, 2**54 + 17]),
        earliest_episode_s=torch.tensor([0., 1_000_000.01], dtype=torch.float64),
        rise_m=torch.tensor([0., .2]), support_height_m=torch.tensor([0., .4]))
    landing = PlatformLandingState(torch.tensor([-1, 2**54 + 17]), torch.tensor([0, 3]),
        torch.tensor([0, int(PlatformLandingPhase.TARGET_SINGLE_WHEEL)]))
    return PlatformRecordState(intent, landing, torch.ones(2, 1, dtype=torch.bool))


def platform_interval():
    flags = torch.tensor([[[False, True, False, True, True, False, False, False, True]] * 4,
                          [[True, True, True, False, False, False, False, False, False],
                           [True, False, False, False, False, False, False, False, False],
                           [True, True, False, True, False, True, False, False, False],
                           [True, True, False, True, True, True, True, False, True]]])
    flags[0, :2, -1] = False
    height = torch.tensor([[torch.nan, torch.nan, .4, .4],
                           [torch.nan, torch.nan, torch.nan, .6]])
    return PlatformLandingInterval(PlatformLandingEvidence(
        flags[..., 0], flags[..., 1], flags[..., 2], flags[..., 3:5], flags[..., 5:7],
        flags[..., 7], flags[..., 8], height), PlatformLandingState(
            torch.tensor([[-1] * 4, [2**54 + 17] * 4]), torch.tensor([[0] * 4, [1, 1, 1, 1]]),
            torch.tensor([[0] * 4, [0, 5, 5, 5]])))


def test_interval_codec_preserves_actual_ticks_world_height_and_unknowns():
    interval = platform_interval()
    record = encode_platform_interval(interval, count=2, device='cpu')
    decoded = decode_platform_interval(record)
    assert PLATFORM_RECORD_VERSION == 2
    assert record['platform_evidence_ticks'].shape == (2, 4, 9)
    assert record['platform_evidence_ticks'][1].tolist() == [
        [True, True, True, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False, False],
        [True, True, False, True, False, True, False, False, False],
        [True, True, False, True, True, True, True, False, True]]
    for name, expected in vars(interval.evidence).items():
        torch.testing.assert_close(getattr(decoded.evidence, name), expected, rtol=0, atol=0, equal_nan=True)
    for name, expected in vars(interval.state).items():
        torch.testing.assert_close(getattr(decoded.state, name), expected, rtol=0, atol=0)
    record['platform_landing_state_ticks'].zero_()
    record['platform_evidence_ticks'].zero_()
    assert decoded.state.request_id[1, 0] == 2**54 + 17
    assert decoded.evidence.full_body_clear[1, 0]
    with pytest.raises(ValueError, match='cannot be inferred'):
        decode_platform_interval({'ticks': torch.zeros(2, 4, 56)})
    with pytest.raises(ValueError, match='version'):
        decode_platform_interval(record, version=1)


class LandingOwner:
    def __init__(self, native, live):
        self.native, self.live = native, live
        self.pending, self.begins, self.aborts = None, 0, 0

    def capture(self):
        return self.live.landing

    def begin_interval(self):
        assert self.pending is None
        self.begins += 1
        self.pending = platform_interval()

    def capture_interval(self):
        assert self.pending is not None
        result, self.pending = self.pending, None
        return result

    def abort_interval(self):
        self.aborts += 1
        self.pending = None


def boundary():
    result = {name: torch.ones(2, dtype=torch.bool) for name in ('valid', 'policy_owned', 'end_policy_owned')}
    result.update({name: torch.zeros(2, dtype=torch.bool) for name in ('terminated', 'truncated')})
    result.update({name: torch.zeros(2, dtype=torch.int64) for name in
                   ('start_episode_id', 'end_episode_id', 'start_control_epoch', 'end_control_epoch')})
    return result


def test_platform_opt_in_does_not_change_old_motion_contract():
    spec = RecordSpec(16, 65)
    assert spec.fields() == spec.fields(record_platform=False)
    before = spec.fields(**OPTIONS)
    after = spec.fields(**OPTIONS, record_platform=True)
    assert {key: value for key, value in after.items() if key not in PLATFORM_RECORD_FIELDS} == before
    assert after['ticks'] == before['ticks'] == (4, TICK_LABEL_DIM) == (4, 56)
    with pytest.raises(ValueError, match='complete jump'):
        spec.fields(record_platform=True)


def test_inactive_and_unknown_are_distinct_and_snapshots_do_not_alias():
    state = platform_state()
    native = SimpleNamespace(supermimic_platform_intent=SimpleNamespace(capture=state.intent.record))
    captured = PlatformRecordState.capture(native, lambda: state.landing, count=2, device='cpu')
    record = captured.record()
    assert captured.valid.all() and captured.intent.phase[0] == PlatformIntentPhase.INACTIVE
    unknown = {**record, 'platform_record_valid': torch.tensor([[False], [True]])}
    assert not PlatformRecordState.decode(unknown).valid[0]
    with pytest.raises(ValueError, match='unknown, not inactive'):
        PlatformRecordState.decode({})
    with pytest.raises(ValueError, match='version'):
        PlatformRecordState.decode(record, version=1)
    state.landing.request_id.fill_(-1)
    state.intent.bound_request_id.fill_(-1)
    assert captured.landing.request_id[1] == 2**54 + 17
    assert captured.intent.bound_request_id[1] == 2**54 + 17
    record['platform_landing_ids'].zero_()
    assert captured.landing.request_id[1] == 2**54 + 17


@pytest.mark.parametrize('missing', ['caller', 'provider', 'wrong_result'])
def test_capture_never_substitutes_empty_for_missing_history(missing):
    state = platform_state()
    native = SimpleNamespace(supermimic_platform_intent=SimpleNamespace(capture=state.intent.record))
    capture = lambda: state.landing
    if missing == 'caller':
        del native.supermimic_platform_intent
    elif missing == 'provider':
        capture = None
    else:
        capture = lambda: None
    with pytest.raises(ValueError):
        PlatformRecordState.capture(native, capture, count=2, device='cpu')


def test_replay_roundtrip_and_sampling_preserve_ids_clocks_unknown_and_false_contacts():
    fields = {'frame': (45,), **PLATFORM_RECORD_FIELDS}
    replay = CompactReplay(fields, 2, 4, 'cpu')
    record = {'frame': torch.zeros(2, 45), **platform_state().record(),
              'contact_valid_ticks': torch.tensor([[[True], [False], [True], [False]]]).expand(2, -1, -1),
              **encode_platform_interval(platform_interval(), count=2, device='cpu')}
    record['platform_record_valid'][0] = False
    for step in range(3):
        replay.add(record, boundary())
    payload = replay.state_dict()
    restored = CompactReplay.from_state_dict(payload)
    audited = CompactReplay.from_state_dict(payload, read_only=True)
    for name, expected in payload['storage'].items():
        torch.testing.assert_close(restored.data[name][:3], expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(audited.data[name], expected, rtol=0, atol=0, equal_nan=True)
    sampled = restored.sample_windows(4, 2, generator=torch.Generator().manual_seed(9))
    for name, dtype in PLATFORM_RECORD_DTYPES.items():
        assert sampled[name].dtype == dtype
    assert (sampled['platform_intent_ids'][..., 2] == 2**54 + 17).any()
    assert (sampled['platform_intent_clock'] == 1_000_000.01).any()
    assert not sampled['contact_valid_ticks'][..., 1, :].any()
    assert torch.isnan(sampled['platform_landing_world_height_ticks'][..., :2, :]).all()
    assert sampled['valid'].all()  # Unknown support height does not invalidate the transition.
    with pytest.raises(ValueError, match='exact shape'):
        replay.add({key: value for key, value in record.items() if key != 'platform_record_valid'}, boundary())
    with pytest.raises(ValueError, match='native replay dtype'):
        replay.add({**record, 'platform_landing_ids': record['platform_landing_ids'].float()}, boundary())
    malformed = {**payload, 'storage': {**payload['storage'],
        'platform_record_valid': payload['storage']['platform_record_valid'].float()}}
    with pytest.raises(ValueError, match='shape/dtype'):
        CompactReplay.from_state_dict(malformed, read_only=True)
    with pytest.raises(ValueError, match='complete typed'):
        CompactReplay({'frame': (45,), 'platform_landing_ids': (3,)}, 2, 3, 'cpu')
    legacy = CompactReplay({'frame': (45,)}, 2, 3, 'cpu')
    assert not set(legacy.data).intersection(PLATFORM_RECORD_FIELDS)


def packed_transition():
    """Valid compact inputs with native substep layout, without a simulator."""
    joints = (*VELOCITY_JOINTS, *PASSIVE_KNEE_JOINTS)
    shapes = {
        'root_orientation_xyzw': (4,), 'course.root_position_local_m': (3,),
        'root_linear_velocity_body_m_s': (3,), 'root_angular_velocity_body_rad_s': (3,),
        'gimbal.yaw_sin_cos': (2,), 'joint_position_gimbal_yaw_sine': (len(joints),),
        'gimbal.pitch_position_rad': (1,), 'joint_velocity_native': (len(joints),),
        'root_height_above_ground_m': (1,), 'course.terrain_heights_relative_m': (77,),
        'identified.wheel_torque_history_nm': (3, 2), 'control._previous_action': (6,),
        'gimbal.controller.target_angles.pitch_rad': (1,),
        'gimbal.controller.target_angles.yaw_sin_cos': (2,),
        'gimbal.controller.yaw_rate_target': (), 'gimbal.controller.mode': (),
        'sensor.values': (3, 18), 'sensor.gimbal_history': (3, 5), 'sensor.heading': (3, 2),
        'sensor.count': (), 'course.dynamic_goal': (), 'course.imagination_remaining_s': (),
        'failure.contact_time_s': (), 'failure.pose_time_s': (), 'failure.failure_reasons': (),
        'course.body_contact_allowed': (), 'course.command_mode': (),
        'course.terrain_tile': (), 'course.tasks.family': (),
    }
    context_shapes = {
        'body_mass_kg': (len(CONTACT_BODIES),), 'requested.motor_strength': (6,),
        'requested.pd_encoder_bias_rad': (6,), 'wheel_material': (4,),
        'identified.wheel_total_inertia_kg_m2': (2,),
        'body_com_pose_local_xyzw': (len(CONTACT_BODIES), 7),
        'body_inertia_kg_m2': (len(CONTACT_BODIES), 9), 'environment_origin_world_m': (3,),
        'requested.policy_encoder_bias_rad': (6,), 'requested.gyro_bias_rad_s': (3,),
        'requested.imu_sensor_to_body_rotation': (9,), 'gimbal.control_joint_signs': (2,),
        'gimbal.joint_zero_rad': (2,), 'sensor.lags': (4,), 'sensor.nominal': (),
        'sensor.noise_scale': (), 'sensor.noise_halfwidth': (18,), 'controller.nominal': (6,),
    }
    fields = tuple(Field(name, shape, frame='native joint order: ' + ','.join(joints))
                   for name, shape in shapes.items())
    schema = StateSchema(fields, tuple(Field(k, v) for k, v in context_shapes.items()),
        observation_dim=45, goal_dim=16, substep_fields=substep_fields(CONTACT_BODIES, joints),
        substeps_per_step=4)
    state = {k: torch.zeros(2, *v) for k, v in shapes.items()}
    state['root_orientation_xyzw'][:, -1] = 1.
    state['gimbal.yaw_sin_cos'][:, 1] = 1.
    state['gimbal.controller.target_angles.yaw_sin_cos'][:, 1] = 1.
    context = {k: torch.ones(2, *v) for k, v in context_shapes.items()}
    endpoint = SimpleNamespace(state=schema.pack(state), context=schema.pack(context, context=True),
        actor_observation=torch.zeros(2, 45), goal=torch.zeros(2, 16), estimator_target=None, estimator_valid=None)
    ticks = {field.name: torch.zeros(2, 4, *field.shape) for field in schema.substep_fields}
    ticks['root_link_orientation_xyzw'][..., -1] = 1.
    ticks['contact_measurement_valid'][0, :, 0] = torch.tensor([1., 0., 1., 0.])
    ticks['contact_measurement_valid'][1, :, 0] = torch.tensor([0., 1., 0., 1.])
    return schema, SimpleNamespace(start=endpoint, end=endpoint, action=torch.zeros(2, 6),
        reward=torch.zeros(2), substeps=schema.pack_substeps(ticks))


def test_layout_retains_every_native_contact_valid_tick_without_changing_motion():
    schema, transition = packed_transition()
    old = SnapshotLayout(schema, record_terrain=True).encode(transition)
    recorded = SnapshotLayout(schema, record_terrain=True, record_platform=True).encode(transition)
    assert recorded['contact_valid_ticks'].dtype == torch.bool
    assert recorded['contact_valid_ticks'].tolist() == [
        [[True], [False], [True], [False]], [[False], [True], [False], [True]]]
    assert set(recorded) == {*old, 'contact_valid_ticks'}
    for name, value in old.items():
        torch.testing.assert_close(recorded[name], value, rtol=0, atol=0)
    assert recorded['ticks'].shape == (2, 4, 56)


@pytest.mark.parametrize('change', ['absent', 'shape', 'substeps'])
def test_layout_rejects_absent_or_incompatible_contact_truth(change):
    schema, _ = packed_transition()
    if change == 'absent':
        schema = replace(schema, substep_fields=tuple(f for f in schema.substep_fields
                                                     if f.name != 'contact_measurement_valid'))
    elif change == 'shape':
        schema = replace(schema, substep_fields=tuple(
            replace(f, shape=(2,)) if f.name == 'contact_measurement_valid' else f for f in schema.substep_fields))
    else:
        schema = replace(schema, substeps_per_step=3)
    SnapshotLayout(schema)  # The legacy contract has not acquired a new requirement.
    with pytest.raises(ValueError, match='four scalar native'):
        SnapshotLayout(schema, record_platform=True)


def collection_fixture(monkeypatch):
    from supermimic_rl.world_model_v2.jump_task import JumpTaskState
    from supermimic_rl.world_model_v2.motion_task import MotionTask
    from supermimic_rl.world_model_v2.stair_task import StairTaskState
    from supermimic_rl.world_model_v2.stair_source import StairSourceState
    from support.world_v2.jump_task import native_jumps

    env = Environment()
    native = native_jumps(2)
    live = platform_state()
    native.supermimic_platform_intent = SimpleNamespace(capture=live.intent.record)
    env.adapter = SimpleNamespace(env=native)
    native.supermimic_platform_landing = LandingOwner(native, live)
    monkeypatch.setattr(MotionTask, 'capture', lambda _: torch.zeros(2, 40))
    monkeypatch.setattr(StairTaskState, 'capture', lambda _: SimpleNamespace(record=lambda: {}))
    monkeypatch.setattr(StairSourceState, 'capture', lambda _: SimpleNamespace(record=lambda: {}))
    monkeypatch.setattr(JumpTaskState, 'capture', lambda _: SimpleNamespace(record=lambda: {}))

    def layout(schema, **options):
        def encode(transition):
            record = {'frame': transition.start.actor_observation}
            if options.get('record_platform'):
                record['contact_valid_ticks'] = transition.contact_valid_ticks.clone()
            return record
        return SimpleNamespace(encode=encode)

    monkeypatch.setattr('supermimic_rl.world_model_v2.collection.SnapshotLayout', layout)
    return env, live


def test_collector_captures_start_before_substeps_and_auto_reset(monkeypatch):
    env, live = collection_fixture(monkeypatch)
    old_step = env.step

    def step(action):
        # Simulate substep progression followed by automatic episode reset.
        live.intent.bound_request_id.fill_(-1)
        live.intent.phase.fill_(int(PlatformIntentPhase.WAITING))
        live.landing.request_id.fill_(-1)
        live.landing.phase.zero_()
        result = old_step(action)
        result.contact_valid_ticks = torch.tensor([[[True], [False], [True], [False]]]).expand(2, -1, -1)
        return result

    env.step = step
    collector = Collector(env, lambda history: (torch.zeros(2, 6), torch.zeros(2, 5)),
        capacity_steps=3, **OPTIONS, record_platform=True, platform_landing=env.adapter.env.supermimic_platform_landing)
    collector.collect(2)
    data = collector.replay.data
    assert data['platform_intent_ids'][0, 1, 2] == 2**54 + 17
    assert data['platform_landing_ids'][0, 1, 0] == 2**54 + 17
    assert data['platform_intent_ids'][1, 1, 2] == -1
    assert data['platform_landing_ids'][1, 1, 0] == -1
    assert data['platform_record_valid'][:2].all()
    assert not data['contact_valid_ticks'][:2, :, 1::2].any()
    assert data['platform_landing_state_ticks'][0, 1, :, 0].tolist() == [2**54 + 17] * 4
    assert data['platform_evidence_ticks'][0, 1, 0, 2]
    assert not data['platform_evidence_ticks'][0, 1, 1, 1]
    assert env.adapter.env.supermimic_platform_landing.begins == 2


def test_collector_missing_capture_and_legacy_replay_are_explicit_errors(monkeypatch):
    env, live = collection_fixture(monkeypatch)
    decide = lambda history: (torch.zeros(2, 6), torch.zeros(2, 5))
    with pytest.raises(ValueError, match='actual landing owner'):
        Collector(env, decide, capacity_steps=3, **OPTIONS, record_platform=True)
    with pytest.raises(ValueError, match='requires opt-in'):
        Collector(env, decide, capacity_steps=3, platform_landing=env.adapter.env.supermimic_platform_landing)
    legacy = CompactReplay({'frame': (45,)}, 2, 3, 'cpu')
    with pytest.raises(ValueError, match='legacy state is not known inactive'):
        Collector(env, decide, capacity_steps=3, **OPTIONS, record_platform=True,
                  platform_landing=env.adapter.env.supermimic_platform_landing, replay=legacy)


def test_collector_cannot_bind_a_different_native_owner(monkeypatch):
    env, live = collection_fixture(monkeypatch)
    foreign = LandingOwner(SimpleNamespace(), live)
    with pytest.raises(ValueError, match='actual landing owner'):
        Collector(env, lambda _: None, capacity_steps=3, **OPTIONS, record_platform=True,
                  platform_landing=foreign)


def test_collector_aborts_partial_record_on_step_failure_without_rolling_back_native(monkeypatch):
    env, live = collection_fixture(monkeypatch)
    owner = env.adapter.env.supermimic_platform_landing

    def fail(action):
        assert owner.pending is not None
        live.landing.phase.fill_(int(PlatformLandingPhase.MISSED))
        raise RuntimeError('physical step interrupted')

    env.step = fail
    collector = Collector(env, lambda _: (torch.zeros(2, 6), torch.zeros(2, 5)),
        capacity_steps=3, **OPTIONS, record_platform=True, platform_landing=owner)
    with pytest.raises(RuntimeError, match='physical step interrupted'):
        collector.collect(1)
    assert owner.pending is None and owner.aborts == 1
    assert collector.steps == 0 and collector.replay is None
    assert live.landing.phase.eq(int(PlatformLandingPhase.MISSED)).all()


def test_diagnostic_nan_height_does_not_enter_motion_fit_or_current_critic():
    from copy import deepcopy
    from support.world_v2.jump_imagination import fixture
    from supermimic_rl.world_model_v2.records import decode_memory, decode_physical

    learner, start, _ = fixture()
    start['end_physical'] = start['physical'].clone()
    windows = {name: value[:, None] for name, value in start.items() if isinstance(value, torch.Tensor)}
    windows['valid'] = torch.ones(2, 1, dtype=torch.bool)
    extra = {**platform_state().record(),
        'contact_valid_ticks': torch.ones(2, 4, 1, dtype=torch.bool),
        **encode_platform_interval(platform_interval(), count=2, device='cpu')}
    recorded = {**windows, **{name: value[:, None] for name, value in extra.items()}}
    assert torch.isnan(recorded['platform_landing_world_height_ticks']).any()
    expected_fit, actual_fit = deepcopy(learner.fit), learner.fit
    expected_fit.observe(windows)
    actual_fit.observe(recorded)
    for name in ('input_norm', 'output_norm'):
        expected = getattr(expected_fit.model.motion, name).state_dict()
        actual = getattr(actual_fit.model.motion, name).state_dict()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    expected_terms, actual_terms = expected_fit.terms(windows), actual_fit.terms(recorded)
    for name in ('state', 'ticks', 'per_step_state', 'terrain_truncated_fraction'):
        torch.testing.assert_close(getattr(actual_terms, name), getattr(expected_terms, name), rtol=0, atol=0)
        assert torch.isfinite(getattr(actual_terms, name)).all()

    def features(value):
        return learner.imagination.critic_features(
            decode_physical(value['physical']), value['height'], value['scan'],
            decode_memory(value['memory']), value['goal'], value['sim_context'],
            value['failure_memory'], value['failure_conditions'],
            torch.cat((value['reward_memory'], value['support_memory']), -1), value.get('motion_task'),
            **learner.imagination.task_features(value))

    plain, extended = features(start), features({**start, **extra})
    assert torch.isfinite(extended).all()
    torch.testing.assert_close(extended, plain, rtol=0, atol=0)
    expected_norm = deepcopy(learner.critic.norm)
    expected_norm.update(plain)
    learner.critic.norm.update(extended)
    for name, value in expected_norm.state_dict().items():
        torch.testing.assert_close(learner.critic.norm.state_dict()[name], value, rtol=0, atol=0)
