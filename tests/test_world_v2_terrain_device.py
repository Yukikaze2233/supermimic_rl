"""Resolve device aliases without weakening atlas provenance/device checks."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

native = pytest.importorskip('transformer_rl.formal.tasks.terrain_query')
from supermimic_rl.world_model_v2.terrain import AtlasTerrain


@pytest.fixture
def pinned_atlas(tmp_path, monkeypatch):
    # The query stub isolates device ownership from Warp/geometry execution.
    # Payload verification still runs through the real atlas loader.
    files = {}
    for name in ('geometry.npz', 'query_index.npz'):
        raw = ('device-test-' + name).encode()
        (tmp_path / name).write_bytes(raw)
        files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    raw = json.dumps({'format': 'terrain_course_atlas_v1', 'files': files,
                      'tiles': [{'origin_m': [1., 2., 3.]}]}).encode()
    path = tmp_path / 'manifest.json'
    path.write_bytes(raw)
    calls = []

    def query(geometry, index, *, device):
        calls.append((geometry, index, device))
        # Match StaticTerrainQuery's actual device assignment, without CUDA.
        return SimpleNamespace(tile_count=1, device=torch.device(device))

    monkeypatch.setattr(native, 'StaticTerrainQuery', query)
    return path, hashlib.sha256(raw).hexdigest(), calls


@pytest.mark.parametrize('device', ['cpu', 'cpu:0', torch.device('cpu:0')])
def test_load_canonicalizes_device_before_native_query_and_tensor_allocation(pinned_atlas, device):
    path, digest, calls = pinned_atlas
    # cpu:0 has the same identity mismatch as cuda vs a concrete CUDA index,
    # so this regression reproduces the production failure entirely on CPU.
    assert torch.device('cpu:0') != torch.empty(0, device='cpu:0').device
    terrain = AtlasTerrain.load(path, expected_sha256=digest, device=device)
    assert len(calls) == 1
    assert calls[0][2] == torch.device('cpu')
    assert terrain.query.device == terrain.origins.device == terrain.offsets.device == torch.device('cpu')
    torch.testing.assert_close(terrain.origins, torch.tensor([[1., 2., 3.]]))
    assert terrain.manifest_sha256 == digest


@pytest.mark.parametrize('origin_device,query_device', [('cpu', 'meta'), ('meta', 'cpu')])
def test_real_device_mismatch_is_still_rejected(origin_device, query_device):
    query = SimpleNamespace(tile_count=1, device=torch.device(query_device))
    origins = torch.empty(1, 3, device=origin_device)
    with pytest.raises(ValueError, match='on the query device'):
        AtlasTerrain(query, origins, manifest_sha256='test')


def test_tile_shape_mismatch_is_still_rejected():
    query = SimpleNamespace(tile_count=2, device=torch.device('cpu'))
    with pytest.raises(ValueError, match='cover every native tile'):
        AtlasTerrain(query, torch.zeros(1, 3), manifest_sha256='test')


@pytest.mark.parametrize('mismatch', ['manifest', 'geometry.npz', 'query_index.npz'])
def test_alias_fix_never_bypasses_provenance_checks(pinned_atlas, mismatch):
    path, digest, calls = pinned_atlas
    if mismatch == 'manifest':
        digest = '0' * 64
    else:
        (path.parent / mismatch).write_bytes(b'changed')
    with pytest.raises(ValueError, match='different atlas manifests|payload differs'):
        AtlasTerrain.load(path, expected_sha256=digest, device='cpu:0')
    assert not calls
