"""Stage relocatable copies without SSH, credentials, torch or GPU startup."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

import pytest


@pytest.fixture
def release_source(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/stage_course_release.py"
    spec = importlib.util.spec_from_file_location("stage_course_release", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    for directory in ("src/package/__pycache__", "scripts", "configs/gpufree", "configs/identified/model", "assets"):
        (root / directory).mkdir(parents=True)
    (root / "src/package/model.py").write_text("VERSION = 1\n")
    (root / "src/package/__pycache__/model.pyc").write_bytes(b"cache")
    (root / "scripts/run_gpufree.sh").write_text("#!/bin/sh\nexit 99\n")
    (root / "configs/identified/model/wheel.json").write_text('{"identified": true}\n')
    (root / "assets/large.usd").write_text("external asset stays outside the release")
    external = "/external/unchanged"
    runtime = {"asset_directory": external + "/vehicle", "terrain_atlas_pin": {
        "path": external + "/terrain/manifest.json", "sha256": "b" * 64}}
    config = root / "configs/gpufree"
    (config / "runtime.json").write_text(json.dumps(runtime))
    (config / "course.json").write_text('{"stage_transition_budgets": [1, 2]}')
    for count in (16, 4096):
        factory = {"runtime_json": "/old/project/configs/gpufree/runtime.json", "runtime_sha256": "old",
                   "course_json": "/old/project/configs/gpufree/course.json", "course_sha256": "old",
                   "identified_bundle": "/old/project/configs/identified/model", "num_envs": count,
                   "upstream_root": external + "/source", "expected_commit": "a" * 40,
                   "asset_directory": external + "/vehicle", "asset_manifest_sha256": "c" * 64,
                   "native_log_directory": "/old/run/native"}
        (config / f"factory{count}.json").write_text(json.dumps(factory))
    for name in ("train-full4096.json", "train-pipeline16.json"):
        (config / name).write_text('{"run": {"updates": 2}, "learning": {}}')
    return module, root


def test_release_rebinds_only_copied_inputs_and_records_independent_code(release_source, tmp_path, monkeypatch):
    module, source = release_source
    monkeypatch.setenv("UNRELATED_SECRET", "not-a-release-field")
    destination = tmp_path / "release with spaces"
    original = (source / "configs/gpufree/factory4096.json").read_bytes()
    receipt = module.stage_release(source, destination)
    factory = json.loads((destination / "configs/gpufree/factory4096.json").read_text())
    assert factory["runtime_json"] == str(destination / "configs/gpufree/runtime.json")
    assert factory["identified_bundle"] == str(destination / "configs/identified/model")
    for name in ("runtime", "course"):
        assert factory[name + "_sha256"] == hashlib.sha256(Path(factory[name + "_json"]).read_bytes()).hexdigest()
    assert factory["asset_directory"] == "/external/unchanged/vehicle"
    assert factory["upstream_root"] == "/external/unchanged/source"
    assert "native_log_directory" not in factory
    assert (source / "configs/gpufree/factory4096.json").read_bytes() == original
    assert not (destination / "assets").exists()
    assert not list(destination.rglob("__pycache__"))
    assert "not-a-release-field" not in (destination / "release.json").read_text()
    argv = shlex.split(receipt["launch_command"])
    assert argv == receipt["profiles"]["selected"]["argv"]
    assert argv[1] == f"SUPERMIMIC_ROOT={destination}"
    assert argv[argv.index("--config") + 1].endswith("train-full4096.json")
    assert argv[argv.index("--factory") + 1] == "supermimic_rl.course.factory:create_course"
    (source / "src/package/model.py").write_text("VERSION = 2\n")
    copied = destination / "src/package/model.py"
    assert copied.read_text() == "VERSION = 1\n"
    assert receipt["code_files_sha256"]["src/package/model.py"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert receipt["external_inputs"]["terrain_atlas_pin"]["path"] == "/external/unchanged/terrain/manifest.json"


@pytest.mark.parametrize("count", [16, 32])
def test_pipeline_profiles_are_release_local(release_source, tmp_path, count):
    module, source = release_source
    destination = tmp_path / f"pipeline-{count}"
    receipt = module.stage_release(source, destination, num_envs=count, pipeline=True)
    profile = receipt["profiles"]["selected"]
    assert profile["training_config"].endswith(f"train-pipeline{count}.json")
    assert json.loads(Path(profile["factory_config"]).read_text())["num_envs"] == count
    assert (destination / "configs/gpufree/train-pipeline32.json").read_bytes() == (
        source / "configs/gpufree/train-pipeline16.json").read_bytes()
    assert set(receipt["profiles"]) == {"selected", "pipeline16", "pipeline32"}


def test_existing_destination_and_recursive_copy_are_rejected_without_mutation(release_source, tmp_path):
    module, source = release_source
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        module.stage_release(source, destination)
    assert list(destination.iterdir()) == [marker] and marker.read_text() == "keep"
    nested = source / "src/release"
    with pytest.raises(ValueError, match="nested"):
        module.stage_release(source, nested)
    assert not nested.exists()
