"""Chronological source proofs preserve original teacher and data identities."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import pytest
import torch

from supermimic_rl.cli import _native_schema_source
from supermimic_rl.contracts import StateSchema
from supermimic_rl.course.recovery_bridge import receipt as bridge_receipt
from supermimic_rl.course.stair_label import install_tof_stair_label
from supermimic_rl.course.stair_retry import timeout_retry_receipt
from supermimic_rl.world_model_v2.auxiliary_data import AuxiliaryWorldData
from supermimic_rl.world_model_v2.migrations.course import verify_course_sampling
from supermimic_rl.world_model_v2.migrations.recovery_bridge import verify_recovery_bridge_sampling
from supermimic_rl.world_model_v2.migrations.source_lineage import (
    compatible_source_lineage, source_lineage_from_saved,
)
from supermimic_rl.world_model_v2.migrations.stair_hint import verify_stair_hint
from supermimic_rl.world_model_v2.migrations.stair_retry import verify_stair_timeout_retry
from supermimic_rl.world_model_v2.stair_source import StairScenario
from supermimic_rl.world_model_v2.teacher import (
    export_ppo_teacher_payload, file_sha256, load_native_teacher,
)
from support.world_v2.stair_source import source_fixture
from support.world_v2.stair_task import native_stairs
from test_world_v2_auxiliary_data import dataset, windows
from test_world_v2_course_sampling_upgrade import candidate
from test_world_v2_recovery_bridge_upgrade import equal_nested
from test_world_v2_native_ppo_teacher import fixture as teacher_fixture


ROOT = Path(__file__).resolve().parents[1]
KEYS = {'course-sampling-only-v1': 'course_sampling',
        'stair-request-hint-lifetime-v1': 'stair_hint',
        'stair-timeout-retry-v1': 'stair_timeout_retry',
        'recovery-roll-bridge-sampling-v1': 'recovery_roll_bridge_sampling'}


def chain(candidate, order=('retry', 'bridge'), extra_source=None):
    saved, _, target, courses = candidate
    before, after = json.loads(saved['schema']['source']), json.loads(target.source)
    native = native_stairs(2, labels=True, assist=True)
    native.cfg.stair_height_assist_spec = replace(native.cfg.stair_height_assist_spec, hold_s=2.)
    native._terrain_course.stair_requests.config = replace(
        native._terrain_course.stair_requests.config, timeout_s=6.)
    _, native._terrain_course.tof_trigger = source_fixture(2)
    old_hint = install_tof_stair_label(native)
    old_scenario = StairScenario.capture(native).to_dict()
    runtime = json.dumps({'stair_height_assist': old_scenario['task']['height_assist']}).encode()
    common = {**(extra_source or {}), 'runtime_sha256': hashlib.sha256(runtime).hexdigest(),
              'stair_label': old_hint}
    before, after = [{**source, **common} for source in (before, after)]
    proofs = [verify_course_sampling(before, after, **courses)]
    sources = [before, after]
    latest = {**after, 'stair_label': install_tof_stair_label(native, height_hint_lifetime='request')}
    scenario = StairScenario.capture(native).to_dict()
    proofs.append(verify_stair_hint(after, latest, old_scenario, scenario, runtime_raw=runtime))
    sources.append(latest)
    course_raw = courses['target_course']
    for action in order:
        if action == 'bridge':
            course = deepcopy(latest['curriculum'])
            course['base']['recovery_roll_bridge'] = True
            raw = json.dumps(course, indent=2).encode()
            following = {**latest, 'course_sha256': hashlib.sha256(raw).hexdigest(),
                         'curriculum': course, 'recovery_roll_bridge': bridge_receipt()}
            proofs.append(verify_recovery_bridge_sampling(latest, following,
                source_course=course_raw, target_course=raw))
            course_raw = raw
        else:
            following = {**latest, 'stair_timeout_retry': timeout_retry_receipt()}
            next_scenario = replace(StairScenario.from_dict(scenario), timeout_retry=True).to_dict()
            proofs.append(verify_stair_timeout_retry(latest, following, scenario, next_scenario,
                runtime_raw=runtime, course_raw=course_raw))
            scenario = next_scenario
        latest = following
        sources.append(latest)
    schema = replace(StateSchema.from_dict(saved['schema']), source=_native_schema_source(latest))
    entries = [{'migration': p['format'], 'cycle': saved['cycle'], KEYS[p['format']]: p} for p in proofs]
    completed = {**saved, 'schema': schema.to_dict(), 'schema_fingerprint': schema.fingerprint,
                 'resume_migrations': [{'old': 'unrelated receipt'}, *entries, {'imagination_horizon': 32}]}
    return completed, schema, tuple(proofs), sources


@pytest.mark.parametrize('order', [('retry', 'bridge'), ('bridge', 'retry')])
def test_two_source_orders_reverse_the_actual_chain_and_stop_at_artifact(candidate, order):
    saved, schema, proofs, sources = chain(candidate, order)
    original = deepcopy(saved)
    assert source_lineage_from_saved(saved, schema) == proofs
    for source in sources:
        assert compatible_source_lineage(source, sources[-1], proofs) == source
    equal_nested(saved, original)
    with pytest.raises(ValueError, match='complete after'):
        compatible_source_lineage(sources[0], sources[-1], (*proofs[:-2], proofs[-1], proofs[-2]))


def test_legacy_teacher_can_keep_new_hint_and_original_course_without_weakening_new_proofs(candidate):
    saved, schema, proofs, sources = chain(candidate)
    teacher_source = {**sources[0], 'stair_label': sources[2]['stair_label']}
    assert compatible_source_lineage(teacher_source, sources[-1], proofs) == teacher_source
    assert source_lineage_from_saved(saved, schema) == proofs
    forged = deepcopy(proofs[-1])
    forged['after']['unregistered_physics'] = 'changed'
    forged['before']['unregistered_physics'] = 'changed'
    # The new proof is internally consistent, but is disconnected from live.
    with pytest.raises(ValueError, match='complete after'):
        compatible_source_lineage(teacher_source, sources[-1], (*proofs[:-1], forged))


@pytest.mark.parametrize('change', ['extra', 'raw', 'before', 'kind', 'cycle', 'wrong_order', 'unknown'])
def test_ledger_rejects_forged_extra_disconnected_and_misordered_proofs(candidate, change):
    saved, schema, proofs, _ = chain(candidate)
    saved = deepcopy(saved)
    entry = saved['resume_migrations'][-2]
    key = KEYS[entry['migration']]
    if change == 'extra':
        entry[key]['unreviewed'] = True
    elif change == 'raw':
        entry[key]['target_course_utf8'] += ' '
    elif change == 'before':
        entry[key]['before']['runtime_sha256'] = 'unrelated'
    elif change == 'kind':
        entry['migration'] = 'stair-timeout-retry-v1'
    elif change == 'cycle':
        entry['cycle'] = saved['cycle'] + 1
    elif change == 'wrong_order':
        saved['resume_migrations'][-2], saved['resume_migrations'][-3] = (
            saved['resume_migrations'][-3], saved['resume_migrations'][-2])
    else:
        entry['migration'] = 'unknown-future-handoff'
    with pytest.raises(ValueError):
        source_lineage_from_saved(saved, schema)


def test_missing_or_unknown_proof_never_authorizes_a_source_difference(candidate):
    saved, schema, proofs, sources = chain(candidate)
    with pytest.raises(ValueError, match='complete after'):
        compatible_source_lineage(sources[0], sources[-1], proofs[:-1])
    unknown = {'format': 'user-provided-source-override', 'before': sources[0], 'after': sources[-1]}
    with pytest.raises(ValueError, match='Unknown'):
        compatible_source_lineage(sources[0], sources[-1], (unknown,))
    saved['resume_migrations'] = [{'migration': 'unregistered', 'anything': unknown}]
    assert source_lineage_from_saved(saved, schema) == ()
    assert compatible_source_lineage(sources[0], sources[-1], ()) != sources[0]


def test_reader_maps_model_only_and_keeps_legacy_course_api(candidate, tmp_path):
    saved, schema, proofs, _ = chain(candidate)
    script = runpy.run_path(str(ROOT / 'scripts/train_world_v2.py'))
    path = tmp_path / 'checkpoint.pt'
    torch.save(saved, path)
    assert script['read_source_lineage'](None, schema) == ()
    assert script['read_source_lineage'](path, schema) == proofs
    assert script['read_course_migrations'](path, schema) == (proofs[0],)
    assert not (tmp_path / saved['replay_file']).exists()


@pytest.mark.parametrize('artifact_position', ['initial', 'middle', 'legacy_composite'])
def test_teacher_qualification_metadata_and_actions_remain_original(candidate, tmp_path, artifact_position):
    _, checkpoint, evaluation, original = teacher_fixture(tmp_path)
    extra = {k: v for k, v in original.items() if k not in ('runtime_sha256', 'course_sha256')}
    _, _, proofs, sources = chain(candidate, extra_source=extra)
    recorded = {'initial': sources[0], 'middle': sources[-2],
                'legacy_composite': {**sources[0], 'stair_label': sources[2]['stair_label']}}[artifact_position]
    report = json.loads(evaluation.read_text())
    report['native_source'] = recorded
    evaluation.write_text(json.dumps(report))
    payload = export_ppo_teacher_payload(checkpoint, evaluation, ['foundation'])
    path = tmp_path / 'teacher.pt'
    torch.save(payload, path)
    digest = file_sha256(path)
    baseline = load_native_teacher(path, digest, device='cpu', native_provenance=recorded)
    with pytest.raises(ValueError, match='source differ'):
        load_native_teacher(path, digest, device='cpu', native_provenance=sources[-1])
    resumed = load_native_teacher(path, digest, device='cpu', native_provenance=sources[-1], source_lineage=proofs)
    assert resumed.source_metadata == baseline.source_metadata
    assert resumed.source_metadata['qualification']['report']['native_source'] == recorded
    assert resumed.source_metadata['qualified_for'] == ['foundation']
    frames = torch.randn(3, 5, 35)
    torch.testing.assert_close(resumed.action_and_estimate(frames), baseline.action_and_estimate(frames), rtol=0, atol=0)
    assert file_sha256(path) == digest
    with pytest.raises(ValueError, match='complete after'):
        load_native_teacher(path, digest, device='cpu',
            native_provenance={**sources[-1], 'runtime_sha256': 'different'}, source_lineage=proofs)


def test_auxiliary_authenticates_full_source_before_reward_filter_and_keeps_split_rng(candidate, tmp_path):
    _, _, proofs, sources = chain(candidate)
    old, replay, config = dataset(tmp_path)
    recorded_source = deepcopy(sources[0])
    recorded_source['reward_kernel']['kernel_sha256'] = 'older-reward-unused-by-motion-fit'
    old = replace(old, source=_native_schema_source(recorded_source))
    live = replace(old, source=_native_schema_source(sources[-1]))
    path = Path(config.path)
    payload = torch.load(path, weights_only=True)
    payload.update(schema=old.to_dict(), schema_fingerprint=old.fingerprint)
    torch.save(payload, path)
    config = replace(config, sha256=file_sha256(path))
    with pytest.raises(ValueError, match='contract differs'):
        AuxiliaryWorldData((config,), live, replay, 2, seed=19)
    baseline = AuxiliaryWorldData((config,), old, replay, 2, seed=19)
    migrated = AuxiliaryWorldData((config,), live, replay, 2, seed=999, source_lineage=proofs)
    baseline.mix(windows(replay))
    state = baseline.state_dict()
    migrated.load_state_dict(state)
    expected, actual = baseline.mix(windows(replay)), migrated.mix(windows(replay))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(migrated.state_dict(), baseline.state_dict(), rtol=0, atol=0)
    assert migrated.source_schemas == {config.name: old.fingerprint}
    assert migrated.sources[0][0] == config
    assert (actual['action'][actual['action'] >= 0] < 1000).all()
    assert file_sha256(path) == config.sha256
    with pytest.raises(ValueError, match='complete after'):
        AuxiliaryWorldData((config,), replace(live, source=_native_schema_source(
            {**sources[-1], 'asset_manifest_sha256': 'unregistered'})), replay, 2, seed=19, source_lineage=proofs)
