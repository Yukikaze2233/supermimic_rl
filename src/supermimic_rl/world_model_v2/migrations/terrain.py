"""Retain learning state across a verified vertical-triangle winding repair."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ...contracts import StateSchema
from ..records import MODEL_CONTRACT
from .teacher import rebind_teacher_source


def verify_atlas_repair(source_runtime, target_runtime):
    """Authenticate both assets and derive affected tiles from actual triangles."""
    runtimes = [json.loads(value) for value in (source_runtime, target_runtime)]
    before, after = runtimes
    if {**after, 'terrain_atlas_pin': before['terrain_atlas_pin']} != before:
        raise ValueError('Terrain repair cannot change other runtime settings')
    manifests, arrays = [], []
    pins = [runtime['terrain_atlas_pin'] for runtime in runtimes]
    for pin in pins:
        path = Path(pin['path'])
        raw = path.read_bytes()
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('Terrain manifest differs from its runtime pin')
        manifest = json.loads(raw)
        for name, expected in manifest['files'].items():
            if Path(name).name != name:
                raise ValueError('Terrain artifact must be inside its atlas')
            contents = (path.parent / name).read_bytes()
            if len(contents) != expected['bytes'] or hashlib.sha256(contents).hexdigest() != expected['sha256']:
                raise ValueError('Terrain artifact differs: ' + name)
        with np.load(path.parent / 'geometry.npz', allow_pickle=False) as data:
            arrays.append({key: data[key].copy() for key in data.files})
        manifests.append(manifest)
    old, new = arrays
    if (manifests[0]['tiles'] != manifests[1]['tiles'] or old.keys() != new.keys()
            or any(not np.array_equal(old[k], new[k]) for k in old if k != 'triangles')):
        raise ValueError('Winding repair must preserve coordinates, routing and physical materials')
    if old['triangles'].shape != new['triangles'].shape:
        raise ValueError('Winding repair must preserve face identities')
    changed = np.flatnonzero((old['triangles'] != new['triangles']).any(axis=1))
    points = old['vertices_m'][old['triangles'][changed]]
    normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    if (not len(changed) or not (normal[:, 2] == 0).all()
            or not np.array_equal(new['triangles'][changed], old['triangles'][changed][:, [0, 2, 1]])):
        raise ValueError('Only existing vertical triangle winding may change')
    return {
        'source_atlas_sha256': pins[0]['sha256'], 'target_atlas_sha256': pins[1]['sha256'],
        'runtime_pair': {'before': hashlib.sha256(source_runtime).hexdigest(),
                         'after': hashlib.sha256(target_runtime).hexdigest()},
        'changed_tiles': np.unique(old['face_tile'][changed]).tolist(),
        'changed_face_count': len(changed), 'coordinates_and_materials_unchanged': True,
    }


def compatible_terrain_source(recorded, live, migrations):
    """Undo only authenticated terrain identities for old, unchanged-tile data.

    Other differences, including a previous disturbance migration, remain for
    the caller's existing contract comparison. Never normalize arbitrary pins.
    """
    target = deepcopy(live)
    excluded = set()
    for migration in reversed(migrations):
        if recorded.get('terrain_sha256') == target.get('terrain_sha256'):
            break
        if (target.get('terrain_sha256') != migration['target_atlas_sha256']
                or target.get('runtime_sha256') != migration['runtime_pair']['after']
                or target.get('authored_routing', {}).get('atlas_sha256') != migration['target_atlas_sha256']):
            continue
        target['terrain_sha256'] = migration['source_atlas_sha256']
        target['runtime_sha256'] = migration['runtime_pair']['before']
        target['authored_routing']['atlas_sha256'] = migration['source_atlas_sha256']
        excluded.update(migration['changed_tiles'])
    return target, sorted(excluded)


def exclude_changed_tiles(storage, changed_tiles):
    """Return a new validity tensor; preserve the immutable physical records."""
    tile = storage['terrain_tile'][..., 0]
    valid = storage['valid']
    if (tile.shape != valid.shape or valid.dtype != torch.bool
            or not torch.isfinite(tile[valid]).all() or not (tile[valid] == tile[valid].round()).all()):
        raise ValueError('Terrain compatibility needs recorded integer routes on valid intervals')
    affected = torch.isin(tile, torch.tensor(changed_tiles, device=tile.device, dtype=tile.dtype))
    return valid & ~affected, int((valid & affected).sum())


def upgrade_terrain(saved, dataset, reference, *, source_runtime, target_runtime,
                    source_teacher_path, target_teacher_path):
    """Copy metadata/validity only; keep all learned tensors, optimizers and RNGs."""
    old, new = (StateSchema.from_dict(item['schema']) for item in (saved, reference))
    if (saved.get('format') != 'supermimic-online-v2' or saved.get('model_contract') != MODEL_CONTRACT
            or reference.get('format') != 'supermimic-world-v2-dataset-v1'
            or reference.get('model_contract') != MODEL_CONTRACT
            or saved['schema_fingerprint'] != old.fingerprint or reference['schema_fingerprint'] != new.fingerprint
            or dataset['schema_fingerprint'] != old.fingerprint or dataset['cycle'] != saved['cycle']):
        raise ValueError('Terrain migration needs a complete paired checkpoint and native reference')
    repair = verify_atlas_repair(source_runtime, target_runtime)
    before, after = old.to_dict(), new.to_dict()
    source, target = json.loads(before.pop('source')), json.loads(after.pop('source'))
    normalized, tiles = compatible_terrain_source(source, target, (repair,))
    if (before != after or normalized != source or tiles != repair['changed_tiles']
            or source['runtime_sha256'] != repair['runtime_pair']['before']):
        raise ValueError('Terrain migration cannot change other native or observation contracts')
    for a, b in (('stair_scenario', 'stair_source_config'), ('jump_scenario', 'jump_task_config'),
                 ('recovery_scenario', 'recovery_task_config')):
        if saved.get(a) != reference.get(b):
            raise ValueError('Terrain migration cannot change task protocols')
    storage = dataset['replay']['storage']
    valid, excluded = exclude_changed_tiles(storage, tiles)
    replay = {**dataset, 'schema_fingerprint': new.fingerprint,
              'replay': {**dataset['replay'], 'storage': {**storage, 'valid': valid}}}
    if 'schema' in replay:
        replay['schema'] = new.to_dict()
    environment = saved['environment']
    course = environment['course']
    router = course['router']
    base = router['base']
    if base['atlas_sha256'] != source['terrain_sha256']:
        raise ValueError('Saved terrain router differs from its source schema')
    environment = {**environment, 'course': {**course, 'router': {**router,
        'base': {**base, 'atlas_sha256': target['terrain_sha256']}}}}
    teacher = rebind_teacher_source(saved, source, target, source_teacher_path, target_teacher_path)
    report = {'migration': 'terrain-riser-winding-v1', 'cycle': saved['cycle'],
        'terrain_repair': repair, 'source_schema': old.fingerprint, 'target_schema': new.fingerprint,
        'historical_valid_intervals': int(storage['valid'].sum()),
        'changed_tile_intervals_excluded': excluded, 'retained_valid_intervals': int(valid.sum()),
        'sequence_before': dataset['replay']['sequence'].tolist(),
        'scope': 'Physical labels retained, changed-tile historical intervals excluded from sampling; '
                 'all learning tensors, optimizers, normalizers, clocks and RNG states retained.'}
    upgraded = {**saved, 'schema': new.to_dict(), 'schema_fingerprint': new.fingerprint,
                'environment': environment, 'teacher_source': teacher,
                'resume_migrations': [*saved.get('resume_migrations', []), report]}
    return upgraded, replay, report
