"""The reward audit includes recorded jumps and credits actual physical evidence."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic_rl.reward_kernel import RewardExtensions
from supermimic_rl.world_model_v2.jump_task import JumpTaskState
from supermimic_rl.world_model_v2.model import Ticks
from supermimic_rl.world_model_v2.records import CONTACT_BODIES, WHEEL_BODIES, decode_memory, decode_physical
from supermimic_rl.world_model_v2.reward_assembly import RewardMemory
from support.world_v2.jump_imagination import fixture


ROOT = Path(__file__).resolve().parents[1]


def audit_fixture(tmp_path, monkeypatch, *, landing=False):
    spec = importlib.util.spec_from_file_location('recorded_jump_reward_audit', ROOT / 'scripts/check_reward_assembly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    learner, start, _ = fixture(gated=True)
    producer = learner.imagination.reward
    producer.jump_task.config = replace(producer.jump_task.config, flight_latch=True)
    start['jump_flight_latch'] = torch.tensor([[-1, 0, 0], [-1, 0, 0]], dtype=torch.int64)
    producer.extensions = RewardExtensions(jump_missed=1., success_event=5.)
    producer.tuning = replace(producer.tuning, success_event=5.)
    state = decode_physical(start['physical'])
    jump = JumpTaskState.decode(start)
    jump.ids[0, 1] = jump.ids[0, 4] = 7
    jump.values[0, [1, 2, 9, 10, 11]] = 0.
    jump.values[0, 5] = .10
    jump.values[0, 6] = .35
    jump.values[0, 8] = 3.
    start['goal'][0, 4] = 0.
    start['goal'][0, 8] = 1.
    start['reward_route'][0, 0] = 4.
    start['task_family'][1] = 0.
    if landing:
        jump.flight_latch[:] = torch.tensor([7, 1, 1])
        jump.ids[:, 1] = jump.ids[:, 4] = 7
        jump.values[:, [1, 9]] = 0.
        jump.values[:, [2, 10, 11, 12, 13]] = 1.
        jump.values[:, 5] = jump.values[:, 15] = .25
        jump.values[:, 6] = .35
        jump.values[:, 7] = 0.
        jump.values[:, 16] = 1.99
        start['goal'][:, 4] = 0.
        start['goal'][:, 8] = 1.
        start['reward_route'][:, 0] = 4.
        start['task_family'].fill_(4.)
    # Prescribed stationary physical samples; no model output creates these labels.
    labels = start['ticks']
    labels.zero_()
    labels[..., 26:28] = 80.
    for name in WHEEL_BODIES:
        labels[..., 28 + CONTACT_BODIES.index(name)] = 80.
    labels[..., 49:51] = 80.
    for index, side in enumerate(producer.geometry.sides):
        q = state.joint_position
        geometry, _ = producer.geometry.linkage.geometry_base(side.name, q[:, side.primary],
                                                               q[:, side.secondary] - q[:, side.primary])
        labels[..., -4 + index] = geometry['passive_joint'][:, side.passive_knee, None]
    ticks = Ticks.decode(labels, state)
    context = producer.bind(start, state, known_interval=True)
    memory = RewardMemory.decode(torch.cat((start['reward_memory'], start['support_memory']), -1))
    result = context.integrate(state, SimpleNamespace(ticks=ticks, tick_terrain_valid=None), start['action'], memory,
        decode_memory(start['memory']), start['goal'], 0, jump_state=jump, base_mode=torch.zeros(2, dtype=torch.long))
    terminated = torch.tensor([False, landing])
    start['reward'] = result.finish(terminated, result.success_credited, context.assembly.parameters)[:, None].float()
    boundaries = {key: torch.ones(2, dtype=torch.bool) for key in ('valid', 'policy_owned', 'end_policy_owned')}
    boundaries['terminated'] = terminated
    data = {name: value[None] for name, value in {**start, **boundaries}.items()}
    source = {'asset_manifest_sha256': producer.geometry.linkage.provenance['asset_manifest_sha256'],
        'terrain_sha256': producer.terrain.manifest_sha256,
        'reward_config': {'robot_mass_kg': producer.tuning.robot_mass_kg, 'gamma': producer.tuning.gamma,
            'profile': 'rough_dash', 'physics_dt_s': .005, 'available_terms': list(producer.terms)},
        'terrain_support_config': asdict(producer.support_config),
        'reward_kernel': {'extensions': asdict(producer.extensions)}}
    schema = SimpleNamespace(source=json.dumps(source), policy_dt_s=.02)
    replay = SimpleNamespace(data=data)
    monkeypatch.setattr(module, 'load_recorded_replay', lambda *args, **kwargs: (replay, schema))
    monkeypatch.setattr(module.AtlasTerrain, 'load', lambda *args, **kwargs: producer.terrain)
    path = tmp_path / 'recorded.pt'
    torch.save({'jump_task_config': asdict(learner.imagination.jump_task.config)}, path)
    arguments = [path, ROOT / 'configs/linkage/leg_linkage_e726d23c.json',
                 ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2', ROOT / 'configs/identified/20261031']
    return module, data, result, arguments


def test_recorded_hold_and_ordinary_subsets_use_true_ticks_and_preserve_default_filter(tmp_path, monkeypatch):
    module, data, result, args = audit_fixture(tmp_path, monkeypatch)
    report = module.check_compact(*args, terrain_path='fixture', recorded_jumps=True)
    assert report['intervals'] == 2 and report['reward_error']['max'] < 2e-6
    assert report['subsets']['ordinary']['intervals'] == 1
    assert report['subsets']['jump']['intervals'] == 1
    assert report['subsets']['hold_without_takeoff']['intervals'] == 1
    assert result.components['cost.jump_missed'][0] == pytest.approx(-.02)
    assert 'not an independent four-reference comparison' in report['reference_scope']
    ordinary = module.check_compact(*args, terrain_path='fixture')
    assert ordinary['intervals'] == 1 and 'subsets' not in ordinary
    data['reward'][0, 0] += .3
    damaged = module.check_compact(*args, terrain_path='fixture', recorded_jumps=True)
    assert damaged['subsets']['hold_without_takeoff']['reward_error']['max'] == pytest.approx(.3, abs=2e-6)
    assert damaged['subsets']['ordinary']['reward_error']['max'] < 2e-6


def test_reward_audit_uses_computed_success_and_suppresses_payment_on_failure(tmp_path, monkeypatch):
    module, _, result, args = audit_fixture(tmp_path, monkeypatch, landing=True)
    assert result.success_credited.tolist() == [True, True]
    report = module.check_compact(*args, terrain_path='fixture', recorded_jumps=True)
    assert report['reward_error']['max'] < 2e-6
    assert report['subsets']['jump']['success_events'] == 1
    assert report['subsets']['hold_without_takeoff']['intervals'] == 0
    assert report['subsets']['hold_without_takeoff']['reward_error'] is None


@pytest.mark.parametrize('damage', ['invalid_jump', 'unknown_gate', 'missing_state', 'missing_config', 'no_atlas'])
def test_recorded_jump_mode_requires_recorded_task_evidence(tmp_path, monkeypatch, damage):
    module, data, _, args = audit_fixture(tmp_path, monkeypatch)
    if damage == 'invalid_jump':
        data['jump_record_valid'].zero_()
    elif damage == 'unknown_gate':
        data['jump_trigger_flags'][..., 0].zero_()
    elif damage == 'missing_state':
        data.pop('jump_values')
    elif damage == 'missing_config':
        torch.save({}, args[0])
    with pytest.raises(ValueError):
        module.check_compact(*args, terrain_path=None if damage == 'no_atlas' else 'fixture', recorded_jumps=True)
