"""Side-roll birth variants preserve native resets and actual endpoint labels."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformer_rl.formal.lab_env.recovery')
from transformer_rl.formal.lab_env.recovery import LabRecoveryReset
from transformer_rl.formal.lab_env.settings import load_asset_description
from transformer_rl.formal.tasks.recovery import RECOVERY_NAMES

from supermimic_rl.course.curriculum import CourseCurriculumConfig, SuperMimicCurriculum
from supermimic_rl.course.recovery_bridge import (
    BRIDGE_ROLL_RAD, RECOVERY_VARIANTS, SIMPLE_POSTURES, RecoveryRollBridge,
    install_recovery_roll_bridge, receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def course(enabled=True, n=32, seed=7, *, only_recovery=True, **options):
    weights = list(CourseCurriculumConfig().stage_weights)
    if only_recovery:
        weights[0] = (0, 0, 0, 0, 0, 1)
    config = CourseCurriculumConfig(recovery_postures=RECOVERY_NAMES,
                                   stage_weights=tuple(weights),
                                   recovery_roll_bridge=enabled, **options)
    return SuperMimicCurriculum(n, seed=seed, config=config)


def assert_same_draw(left, right, *, skip=()):
    for field in fields(left):
        if field.name not in skip:
            assert torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name


def test_disabled_has_exact_legacy_config_and_sampler():
    original, candidate = course(False), course(False)
    assert 'recovery_roll_bridge' not in candidate.config.to_dict()
    assert len(candidate.sample_counts['recovery_posture']) == 16
    assert install_recovery_roll_bridge(SimpleNamespace(), candidate, '.') is None
    for _ in range(3):
        assert_same_draw(original.sample(torch.arange(32)), candidate.sample(torch.arange(32)))
    with pytest.raises(ValueError, match='boolean'):
        CourseCurriculumConfig(recovery_roll_bridge=1)


def test_two_cycles_have_six_bridges_without_reducing_difficult_births_or_rng():
    original, candidate = course(False), course()
    left, right = original.sample(torch.arange(32)), candidate.sample(torch.arange(32))
    variants = candidate.recovery_birth_variants(torch.arange(32))
    counts = torch.bincount(variants, minlength=22)
    assert counts[16:].tolist() == [1] * 6
    assert counts[list(SIMPLE_POSTURES)].tolist() == [1] * 6
    difficult = [i for i in range(16) if i not in SIMPLE_POSTURES]
    assert counts[difficult].tolist() == [2] * 10
    assert (right.recovery_posture[variants >= 16] ==
            4 + (variants[variants >= 16] - 16) % 2).all()
    assert_same_draw(left, right, skip=('recovery_posture',))
    assert candidate.sample_counts['recovery_posture'] == counts.tolist()
    assert candidate.sample_counts['family'] == original.sample_counts['family']
    assert torch.equal(candidate.sampler.generator.get_state(), original.sampler.generator.get_state())


def test_mixed_families_and_earlier_partial_pose_pool_are_unchanged():
    original, candidate = course(False, n=256, only_recovery=False), course(n=256, only_recovery=False)
    total = sum(original.stage_budgets)
    original.set_progress(total, 10)
    candidate.set_progress(total, 10)
    for _ in range(4):
        left, right = original.sample(torch.arange(256)), candidate.sample(torch.arange(256))
        assert_same_draw(left, right, skip=('recovery_posture',))
        nonrecovery = left.family != 5
        assert torch.equal(left.recovery_posture[nonrecovery], right.recovery_posture[nonrecovery])
        assert candidate.sample_counts['family'] == original.sample_counts['family']
        assert torch.equal(candidate.sampler.generator.get_state(), original.sampler.generator.get_state())
    options = {'recovery_posture_stages': {'left_side_90': 'full_recovery'}}
    original, candidate = course(False, **options), course(**options)
    for _ in range(3):
        assert_same_draw(original.sample(torch.arange(32)), candidate.sample(torch.arange(32)))
        assert (candidate.recovery_birth_variants(torch.arange(32)) < 16).all()


def test_resume_preserves_next_variant_and_rng_across_irregular_batch_sizes():
    current = course(n=7)
    for ids in ([2], [5, 1], [0, 6, 2, 4], [3, 5, 1]):
        current.sample(ids)
    state = deepcopy(current.state_dict())
    restored = course(n=7, seed=123)
    restored.load_state_dict(state)
    for ids in ([4, 1], [2], [0, 1, 2, 3, 4, 5, 6], [2, 6, 5]):
        assert_same_draw(current.sample(ids), restored.sample(ids))
        assert torch.equal(current.recovery_birth_variants(ids), restored.recovery_birth_variants(ids))
        assert current.sample_counts == restored.sample_counts
    legacy = course(False, n=7)
    with pytest.raises(ValueError, match='identity/configuration'):
        restored.load_state_dict(legacy.state_dict())


def test_bridge_receipt_is_not_aliased_to_old_world_model_source():
    from supermimic_rl.cli import _native_schema_source

    old = {'curriculum': course(False).config.to_dict()}
    new = {'curriculum': course().config.to_dict(), 'recovery_roll_bridge': receipt()}
    assert _native_schema_source(old) != _native_schema_source(new)
    assert json.loads(_native_schema_source(new))['recovery_roll_bridge'] == receipt()


@pytest.fixture(scope='module')
def native_reset():
    path = Path(os.environ.get('SUPERMIMIC_TEST_ASSET',
                              ROOT / 'assets/full_vehicle_v2p0p1_source_audited_v2_knee40_107'))
    if not path.is_dir():
        pytest.skip('The authenticated selected asset is required for placement checks')
    digest = hashlib.sha256((path / 'manifest.json').read_bytes()).hexdigest()
    assert digest == '08bf7e001db99f21ee1e91133bf32fd7567b1db8c75fc31c908608ae85dcc1f8'
    torch.set_num_threads(2)
    return LabRecoveryReset(load_asset_description(path, expected_manifest_sha256=digest), 32)


def bridge(native_reset, tmp_path, *, evaluation=False):
    curriculum = course()
    draw = curriculum.sample(torch.arange(32))
    value = RecoveryRollBridge(SimpleNamespace(_recovery_reset=native_reset), curriculum,
                               tmp_path, evaluation=evaluation)
    return value, draw


def test_compiler_places_each_roll_instead_of_rotating_old_z(native_reset, tmp_path):
    value, draw = bridge(native_reset, tmp_path)
    result = value.pose(torch.arange(32), draw.recovery_posture, torch.zeros(32), torch.zeros(32, 3))
    variant = value.curriculum.recovery_birth_variants(torch.arange(32))
    indices = (variant >= 16).nonzero().flatten()
    expected = torch.tensor(BRIDGE_ROLL_RAD)[variant[indices] - 16]
    torch.testing.assert_close(value.roll(result.root_pose_xyzw[indices, 3:]), expected, atol=2e-7, rtol=0)
    original = value.original_pose(torch.arange(32), draw.recovery_posture, torch.zeros(32), torch.zeros(32, 3))
    assert (result.root_pose_xyzw[indices, 2] - original.root_pose_xyzw[indices, 2]).abs().min() > .001
    unchanged = (variant < 16).nonzero().flatten()
    for field in fields(original):
        assert torch.equal(getattr(result, field.name)[unchanged], getattr(original, field.name)[unchanged])
    assert result.geometry_ready.all()
    assert torch.count_nonzero(result.root_com_velocity[indices]) == 0
    assert torch.count_nonzero(result.joint_velocity[indices]) == 0
    # Independently re-place the written joint ordering/orientation. The old
    # 90-degree z would fail this height comparison for every bridge variant.
    reset = native_reset
    order = [reset.kinematics.binding.native_joint_names.index(name)
             for name in reset.kinematics.binding.tree_joint_names]
    tree = result.joint_position[indices][:, order].double()
    quaternion = result.root_pose_xyzw[indices, 3:].double()
    quaternion /= quaternion.norm(dim=-1, keepdim=True)
    certified = reset.geometry.place(tree, quaternion, torch.zeros(len(indices), 2, dtype=torch.float64),
        reset.flat_patch(indices, torch.zeros(len(indices), 3, dtype=torch.float64)),
        clearance_m=reset.config.clearance_m, loop_tolerance_m=reset.config.loop_tolerance_m,
        wheel_support_tolerance_m=reset.config.wheel_support_tolerance_m)
    assert certified.geometry_ready.all()
    assert certified.body_minimum_gap_m.min() >= reset.config.clearance_m - 1e-8
    torch.testing.assert_close(result.root_pose_xyzw[indices, 2].double(), certified.root_pose_xyzw[:, 2],
                               atol=1e-7, rtol=0)


def test_cache_reuse_and_nonzero_yaw_height_recompile(native_reset, tmp_path, monkeypatch):
    value, draw = bridge(native_reset, tmp_path)
    ids = (value.curriculum.recovery_birth_variants(torch.arange(32)) >= 16).nonzero().flatten()
    posture = draw.recovery_posture[ids]
    calls = []
    original = value.reset.compiler.compile
    def compile(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(value.reset.compiler, 'compile', compile)
    origins = torch.zeros(6, 3)
    origins[:, :2] = torch.arange(12).reshape(6, 2)
    result = value.pose(ids, posture, torch.zeros(6), origins)
    assert not calls
    assert torch.equal(result.root_pose_xyzw[:, :2], origins[:, :2])
    result.root_pose_xyzw.fill_(999)
    again = value.pose(ids, posture, torch.zeros(6), origins)
    assert (again.root_pose_xyzw[:, 2] < 1).all()
    origins[:, 2] = .3
    elevated = value.pose(ids, posture, torch.full((6,), .2), origins)
    assert calls == [6]
    torch.testing.assert_close(elevated.root_pose_xyzw[:, 2], again.root_pose_xyzw[:, 2] + .3,
                               atol=1e-6, rtol=0)


def test_fixed_evaluation_keeps_all_sixteen_original_poses(native_reset, tmp_path):
    value, _ = bridge(native_reset, tmp_path, evaluation=True)
    ids = posture = torch.arange(16)
    before = value.original_pose(ids, posture, torch.zeros(16), torch.zeros(16, 3))
    after = value.pose(ids, posture, torch.zeros(16), torch.zeros(16, 3))
    for field in fields(before):
        assert torch.equal(getattr(before, field.name), getattr(after, field.name))
    assert value.cache is None
    assert [record['variant'] for record in value.pending.values()] == list(RECOVERY_NAMES)


def test_explicit_fixed_bridges_use_real_placement_and_repeat_across_resets(native_reset, tmp_path):
    value, _ = bridge(native_reset, tmp_path, evaluation=True)
    value.fixed_variants = torch.arange(32) % len(RECOVERY_VARIANTS)
    ids = torch.tensor([21, 16, 19, 4, 5, 15])
    labels = value.fixed_variants[ids]
    posture = torch.where(labels >= 16, 4 + (labels - 16) % 2, labels)
    rng = value.curriculum.sampler.generator.get_state().clone()
    first = value.pose(ids, posture, torch.zeros(len(ids)), torch.zeros(len(ids), 3))
    second = value.pose(ids.flip(0), posture.flip(0), torch.zeros(len(ids)), torch.zeros(len(ids), 3))
    for field in fields(first):
        assert torch.equal(getattr(first, field.name), getattr(second, field.name).flip(0))
    torch.testing.assert_close(value.roll(first.root_pose_xyzw[:3, 3:]),
                               torch.tensor([BRIDGE_ROLL_RAD[i - 16] for i in labels[:3]]),
                               atol=2e-7, rtol=0)
    original = value.original_pose(ids[3:], posture[3:], torch.zeros(3), torch.zeros(3, 3))
    for field in fields(first):
        assert torch.equal(getattr(first, field.name)[3:], getattr(original, field.name))
    assert first.geometry_ready.all()
    assert torch.equal(value.curriculum.sampler.generator.get_state(), rng)
    assert [value.pending[int(i)]['variant_id'] for i in ids] == labels.tolist()
    with pytest.raises(ValueError, match='variant and native posture'):
        value.pose(ids[:1], torch.tensor([4]), torch.zeros(1), torch.zeros(1, 3))


def test_trace_distinguishes_variant_request_placement_and_actual_episode(native_reset, tmp_path):
    value, draw = bridge(native_reset, tmp_path)
    pose = value.pose(torch.arange(32), draw.recovery_posture, torch.zeros(32), torch.zeros(32, 3))
    readback = pose.root_pose_xyzw.clone()
    # Simulate readback drift so the audit cannot accidentally label the
    # requested/compiled quaternion as actual physics feedback.
    readback[16, 3:] = torch.tensor([0., 0., 0., 1.])
    readback[16, 2] += .01
    value.record_births(torch.arange(32) + 2, torch.ones(32) * .02, readback, readback[:, 2] + .03)
    records = [json.loads(line) for line in value.trace_path.read_text().splitlines()]
    assert len(records) == 32 and not value.pending
    selected = records[16]
    assert selected['variant'] == 'left_side_45'
    assert selected['native_base_posture_id'] == 4
    assert selected['episode_id'] == 18
    assert selected['requested_roll_rad'] == -math.pi / 4
    assert selected['actual_birth_roll_rad'] == 0
    assert abs(selected['placed_roll_rad'] + math.pi / 4) < 2e-7
    assert selected['actual_root_world_z_m'] > selected['placed_root_world_z_m']
    assert selected['actual_base_com_world_z_m'] > selected['actual_root_world_z_m']
    assert receipt()['variants'][4] == 'left_side_90'
    assert receipt()['variants'][16] == 'left_side_45'
    value.record_births(torch.zeros(32), torch.zeros(32), readback, readback[:, 2] + .03)
    assert len(value.trace_path.read_text().splitlines()) == 32
