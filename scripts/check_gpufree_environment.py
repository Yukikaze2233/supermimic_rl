#!/usr/bin/env python3
"""Inspect a GPUFREE image without importing Isaac, Torch, Newton, or Warp.

The default path reads files/metadata and queries nvidia-smi only. --vulkan
explicitly enumerates Vulkan devices; it still does not launch Isaac or train.
This is a diagnostic report, not a claim of native simulator compatibility.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REFERENCE_VERSIONS = {
    "isaacsim": "6.2.0.0",
    "isaaclab": "28.0.0",
    "isaaclab-newton": "9.1.4",
    "newton": "1.6.1",
    "warp-lang": "1.17.0",
    "torch": "2.11.0+cu130",
}


def command_report(argv: list[str], timeout: int = 15, *, output_limit: int = 24000) -> dict:
    """Run a bounded diagnostic without a shell or installation side effects."""
    executable = shutil.which(argv[0])
    if executable is None:
        return {"command": argv, "status": "not_installed"}
    try:
        result = subprocess.run(
            [executable, *argv[1:]], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "status": "error", "error": str(error)}
    return {
        "command": argv, "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout[-output_limit:], "stderr": result.stderr[-8000:],
    }


def disk_report(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    return {
        "path": str(path), "resolved": str(path.resolve()), "exists": True,
        "is_mount": path.is_mount(), "writable": os.access(path, os.W_OK),
        "total_gib": round(usage.total / 2**30, 2),
        "free_gib": round(usage.free / 2**30, 2),
    }


def package_report() -> dict:
    result = {}
    for name, expected in REFERENCE_VERSIONS.items():
        try:
            distribution = importlib.metadata.distribution(name)
            actual = distribution.version
            result[name] = {
                "installed": actual, "kaiser_reference": expected,
                "exact_metadata_match": actual == expected,
                "location": str(distribution.locate_file("")),
            }
        except importlib.metadata.PackageNotFoundError:
            result[name] = {"installed": None, "kaiser_reference": expected}
    return result


def icd_report() -> list[dict]:
    paths = set()
    for directory in ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d"):
        paths.update(Path(directory).glob("*.json"))
    for key in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES"):
        paths.update(Path(value) for value in os.environ.get(key, "").split(os.pathsep) if value)
    result = []
    for path in sorted(paths):
        try:
            content = json.loads(path.read_text())
            result.append({"path": str(path), "icd": content.get("ICD", {})})
        except (OSError, ValueError) as error:
            result.append({"path": str(path), "error": str(error)})
    return result


def vulkan_loader() -> str | None:
    # Avoid ctypes: a mismatched bundled Python must still produce metadata.
    report = command_report(["ldconfig", "-p"], output_limit=200000)
    for line in report.get("stdout", "").splitlines():
        if "libvulkan.so.1 " in line and "=>" in line:
            return line.split("=>", 1)[1].strip()
    return None


def isaac_report(root: Path) -> dict:
    """Read standalone SDK identity and its own RTX policy, without Kit imports.

    The bundled version-2 policy uses whole-line JSON comments. Preserve each
    architecture/category condition instead of inferring a universal minimum.
    """
    result = {"root": str(root), "exists": root.is_dir()}
    for name in ("VERSION", "kit/driver-requirements.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
            result[name] = {"sha256": hashlib.sha256(content).hexdigest()}
            if name == "VERSION":
                result[name]["value"] = content.decode().strip()
            else:
                policy = json.loads("\n".join(
                    line for line in content.decode().splitlines()
                    if not line.lstrip().startswith("//")
                ))
                result[name]["format_version"] = policy.get("version")
                if policy.get("version") == 2:
                    result[name]["rtx_linux_x86_64"] = policy.get("rtx", {}).get("linux-x86_64")
                else:
                    result[name]["warning"] = "Unknown policy format; no interpretation attempted."
        except (OSError, ValueError, UnicodeError) as error:
            result[name] = {"error": str(error)}
    return result


def python_report() -> dict:
    result = {"executable": sys.executable, "version": platform.python_version()}
    config = Path(sys.prefix) / "pyvenv.cfg"
    if config.is_file():
        result["pyvenv_config"] = {
            key.strip(): value.strip()
            for line in config.read_text().splitlines() if "=" in line
            for key, value in [line.split("=", 1)]
        }
        expected = result["pyvenv_config"].get("version_info")
        if expected is not None and expected != result["version"]:
            result["warning"] = "Python differs from pyvenv.cfg; check bundled libpython resolution via LD_LIBRARY_PATH."
    return result


def factory_report(path: Path) -> dict:
    """Check transport-critical pins, without loading or modifying the factory."""
    try:
        config = json.loads(path.read_text())
        result = {"path": str(path), "checks": {}}
        for field, pin, filename in (
            ("runtime_json", "runtime_sha256", None),
            ("asset_directory", "asset_manifest_sha256", "manifest.json"),
        ):
            target = Path(config[field]).expanduser()
            if filename is not None:
                target = target / filename
            digest = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
            result["checks"][field] = {
                "path": str(target), "expected_sha256": config.get(pin),
                "actual_sha256": digest,
                "matches": digest is not None and digest == config.get(pin),
            }
        source = Path(config["upstream_root"]).expanduser()
        head = command_report(["git", "-C", str(source), "rev-parse", "HEAD"])
        commit = head.get("stdout", "").strip() if head["status"] == "ok" else None
        result["checks"]["upstream_root"] = {
            "path": str(source), "actual_commit": commit,
            "expected_commit": config.get("expected_commit"),
            "matches": commit is not None and commit == config.get("expected_commit"),
        }
        result["scope"] = "Top-level transport pins only; native factory validates full asset/source/runtime contracts."
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"path": str(path), "error": str(error)}


def collect(data_root: Path, *, vulkan: bool = False, factory: Path | None = None,
            isaac_root: Path = Path("/isaac-sim")) -> dict:
    gpu = command_report([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ])
    gpu["devices"] = []
    if gpu["status"] == "ok":
        for values in csv.reader(io.StringIO(gpu["stdout"])):
            if len(values) == 4:
                gpu["devices"].append(dict(zip(
                    ("name", "driver", "memory_mib", "pci_bus_id"),
                    (value.strip() for value in values),
                )))
    packages = package_report()
    isaac = isaac_report(isaac_root)
    warnings = []
    if not gpu["devices"]:
        warnings.append("No GPU detected by nvidia-smi; this may be a CPU-only boot or missing host GPU injection.")
    if not data_root.is_dir():
        warnings.append("The configured persistent data directory does not exist; verify the platform mount before copying data.")
    for name, item in packages.items():
        if item["installed"] is None:
            if name == "isaacsim" and "VERSION" in isaac:
                continue
            warnings.append(f"{name}: no distribution metadata in this Python; check the intended venv/standalone installation.")
        elif not item["exact_metadata_match"]:
            warnings.append(f"{name}: differs from Kaiser reference; do not silently replace the pinned native stack.")
    for item in gpu["devices"]:
        if item["driver"] != "595.58.03":
            warnings.append("Host driver differs from Isaac Sim 6.2 Linux tested version 595.58.03; this is not an automatic incompatibility verdict.")
            break
    result = {
        "format": "supermimic-gpufree-inspection-v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Metadata and nvidia-smi only; no Isaac/Torch/Newton/Warp import or simulation.",
        "python": python_report(), "isaac_standalone": isaac,
        "system": {"platform": platform.platform(), "architecture": platform.machine()},
        "packages": packages, "gpu": gpu,
        "vulkan_loader": vulkan_loader(), "vulkan_icds": icd_report(),
        "device_nodes": sorted(str(p) for pattern in ("nvidia*", "dri/*") for p in Path("/dev").glob(pattern)),
        "storage": [disk_report(p) for p in (Path("/"), data_root, Path("/dev/shm"))],
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
            "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "XDG_CACHE_HOME", "UV_CACHE_DIR",
            "LD_LIBRARY_PATH", "ISAACSIM_ROOT_PATH", "ISAACLAB_PATH",
        )},
        "tools": {key: shutil.which(key) for key in ("tmux", "rsync", "vulkaninfo", "git")},
        "warnings": warnings,
    }
    if vulkan:
        result["vulkan_enumeration"] = command_report(["vulkaninfo", "--summary"], timeout=30)
        result["scope"] += " Explicit --vulkan device enumeration requested."
    if factory is not None:
        result["factory"] = factory_report(factory)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/gpufree-data"))
    parser.add_argument("--isaac-root", type=Path, default=Path(os.environ.get("ISAACSIM_ROOT_PATH", "/isaac-sim")))
    parser.add_argument("--factory-config", type=Path, help="Optional read-only top-level path/pin checks")
    parser.add_argument("--vulkan", action="store_true", help="Explicitly run vulkaninfo --summary")
    parser.add_argument("--output", type=Path, help="Write JSON here; parent directory must already exist")
    args = parser.parse_args(argv)
    report = collect(args.data_root, vulkan=args.vulkan, factory=args.factory_config,
                     isaac_root=args.isaac_root)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
