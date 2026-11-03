"""Report restoration uses saved contact filters and pinned native clock limits."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'report_world_v2_data', Path(__file__).parents[1] / 'scripts/report_world_v2_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_imagination_floors_are_restored_without_refitting():
    saved = {'contact_floor': {'body': torch.arange(21).float(), 'base': torch.tensor(4.)}}
    floors = module.checkpoint_contact_floor(saved, 'cpu')
    torch.testing.assert_close(floors[0], saved['contact_floor']['body'])
    torch.testing.assert_close(floors[1], saved['contact_floor']['base'])
    assert module.checkpoint_contact_floor({}, 'cpu') is None


def test_contact_clock_requires_the_recorded_runtime_pin(tmp_path):
    path = tmp_path / 'runtime.json'
    runtime = {'termination_limits': {
        'minimum_normal_height_m': .1, 'minimum_recovery_height_m': .05,
        'failure_tilt_rad': 1., 'forbidden_body_load_n': 1.,
        'forbidden_contact_confirm_s': .04, 'pose_failure_confirm_s': .04,
        'forbidden_body_names': ['base_link', 'M_link1', 'M_link2']}}
    path.write_text(json.dumps(runtime))
    schema = SimpleNamespace(policy_dt_s=.02, substeps_per_step=4,
        source=json.dumps({'runtime_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    termination = module.pinned_termination(schema, path)
    assert termination.dt == .005 and termination.limits.forbidden_contact_confirm_s == .04
    assert module.pinned_termination(schema, None) is None
    path.write_text(json.dumps(runtime) + '\n')
    with pytest.raises(ValueError, match='SHA256'):
        module.pinned_termination(schema, path)
