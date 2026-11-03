#!/usr/bin/env python3
"""Copy an independent native-course release and print its pinned launch command.

Run this on the training host after synchronizing the working project. It uses
only the Python standard library and never imports or launches the simulator.
Existing destinations are never merged or overwritten. External upstream and
physical asset pins remain unchanged; only copied configuration paths relocate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def copied_configuration(source, value):
    """Resolve a config reference from either the local or deployed project root."""
    path = Path(value).expanduser()
    if path.is_absolute() and path.is_relative_to(source):
        relative = path.relative_to(source)
    elif path.parts and path.parts[0] == "configs":
        relative = path
    elif "configs" in path.parts:
        # Deployed templates can be staged locally for CPU verification, too.
        # Do not rewrite arbitrary absolute paths, especially physical assets.
        relative = Path(*path.parts[path.parts.index("configs"):])
    else:
        raise ValueError("Release-local input must be inside copied configs: " + str(path))
    actual = (source / relative).resolve(strict=True)
    if not actual.is_relative_to(source / "configs"):
        raise ValueError("Configuration reference escapes the copied config tree")
    return relative


def manifest(directory, sections):
    files = {}
    for section in sections:
        for path in sorted((directory / section).rglob("*")):
            if path.is_file():
                files[path.relative_to(directory).as_posix()] = sha256(path)
    return files


def launch_profile(destination, num_envs, training_config, label):
    factory = destination / "configs" / "gpufree" / f"factory{num_envs}.json"
    output = destination / "outputs" / label
    argv = ["env", f"SUPERMIMIC_ROOT={destination}", "bash",
            str(destination / "scripts" / "run_gpufree.sh"), "train",
            "--factory", "supermimic_rl.course.factory:create_course",
            "--config", str(training_config), "--factory-config", str(factory),
            "--output", str(output)]
    return {"num_envs": num_envs, "training_config": str(training_config),
            "factory_config": str(factory), "output": str(output),
            "argv": argv, "command": shlex.join(argv)}


def stage_release(source_root, destination, *, num_envs=4096, pipeline=False):
    source = Path(source_root).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Release destination already exists: " + str(destination))
    destination = destination.resolve()
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if pipeline and num_envs not in (16, 32):
        raise ValueError("Pipeline profiles use --num-envs 16 or 32")
    for section in ("src", "scripts", "configs"):
        if not (source / section).is_dir():
            raise FileNotFoundError(source / section)
        if destination.is_relative_to(source / section):
            raise ValueError("Release destination cannot be nested in a copied source tree")

    configuration = source / "configs" / "gpufree"
    templates = {}
    for path in sorted(configuration.glob("factory*.json")):
        value = json.loads(path.read_text())
        local = {name: copied_configuration(source, value[name])
                 for name in ("runtime_json", "course_json", "identified_bundle")}
        if not (source / local["identified_bundle"]).is_dir():
            raise ValueError("identified_bundle must identify the copied model directory")
        templates[path.name] = (value, local)
    if "factory4096.json" not in templates or "factory16.json" not in templates:
        raise FileNotFoundError("The formal4096 and pipeline16 factory templates are required")
    for filename in ("train-full4096.json", "train-pipeline16.json"):
        if not (configuration / filename).is_file():
            raise FileNotFoundError(configuration / filename)
    if not (source / "scripts" / "run_gpufree.sh").is_file():
        raise FileNotFoundError(source / "scripts" / "run_gpufree.sh")

    # mkdir is the atomic reservation: a racing stager cannot merge a release.
    destination.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for section in ("src", "scripts", "configs"):
        shutil.copytree(source / section, destination / section, ignore=ignore)
    target = destination / "configs" / "gpufree"

    def write_factory(filename, count, template):
        original, local = templates[template]
        value = {**original, "num_envs": count}
        for name, relative in local.items():
            value[name] = str(destination / relative)
        value["runtime_sha256"] = sha256(value["runtime_json"])
        value["course_sha256"] = sha256(value["course_json"])
        # The CLI derives logs from the selected fresh output, never an old run.
        value.pop("native_log_directory", None)
        write_json(target / filename, value)
        return value

    factories = {}
    for filename, (value, _) in templates.items():
        factories[filename] = write_factory(filename, value["num_envs"], filename)
    if "factory32.json" not in factories:
        factories["factory32.json"] = write_factory("factory32.json", 32, "factory16.json")
    selected_factory = f"factory{num_envs}.json"
    if selected_factory not in factories:
        factories[selected_factory] = write_factory(selected_factory, num_envs, "factory4096.json")

    pipeline32 = target / "train-pipeline32.json"
    if not pipeline32.exists():
        # Number of environments belongs to the factory; retain the same short
        # warmup/world-only/online pipeline settings and network architecture.
        shutil.copy2(target / "train-pipeline16.json", pipeline32)
    selected_config = target / (f"train-pipeline{num_envs}.json" if pipeline
                                else f"train-full{num_envs}.json")
    if not selected_config.is_file():
        selected_config = target / "train-full4096.json"
    profiles = {"selected": launch_profile(destination, num_envs, selected_config,
                                          f"{'pipeline' if pipeline else 'full'}{num_envs}")}
    for count in (16, 32):
        profiles[f"pipeline{count}"] = launch_profile(
            destination, count, target / f"train-pipeline{count}.json", f"pipeline{count}")

    code = manifest(destination, ("src", "scripts"))
    configs = manifest(destination, ("configs",))
    selected = factories[selected_factory]
    runtime = json.loads(Path(selected["runtime_json"]).read_text())
    receipt = {
        "format": "supermimic-course-release-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source), "release_root": str(destination),
        "copied_sections": ["src", "scripts", "configs"],
        "code_sha256": hashlib.sha256(json.dumps(code, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "code_files_sha256": code, "configuration_files_sha256": configs,
        "external_inputs": {name: selected[name] for name in (
            "upstream_root", "expected_commit", "asset_directory", "asset_manifest_sha256")},
        "profiles": profiles, "launch_command": profiles["selected"]["command"],
        "scope": "independent code/config copy; external source and physical assets remain pinned; no simulator launched",
    }
    receipt["external_inputs"]["terrain_atlas_pin"] = runtime.get("terrain_atlas_pin")
    write_json(destination / "release.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--pipeline", action="store_true",
                        help="Select the short native pipeline with --num-envs 16 or 32")
    args = parser.parse_args()
    receipt = stage_release(args.source_root, args.destination, num_envs=args.num_envs, pipeline=args.pipeline)
    print(json.dumps({"release": str(Path(receipt["release_root"]) / "release.json"),
                      "code_sha256": receipt["code_sha256"],
                      "launch_command": receipt["launch_command"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
