"""Authenticate the reviewed EXTENSION height lower-bound opt-in."""
from copy import deepcopy

from .reward_continuity import _upgrade_reward_pair


FORMAT = 'jump-extension-height-floor-v1'
SOURCE_KERNEL_SHA256 = '7ab9e5481d13cfc029852aa0ee44c0a3cf4cdb5672b3b269e8878cff4257850e'
TARGET_KERNEL_SHA256 = '459439a4f56c19d3115ed55ad415386129db7a5325a603849fdc0ae8094b3b61'


def verify_jump_extension_height_floor(source, target):
    """Permit exactly one opt-in and its reviewed kernel on the settled source."""
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('Extension height proof requires complete source objects')
    before, after = source.get('reward_kernel'), target.get('reward_kernel')
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or before.get('implementation') != 'shared_physical_reward_densities'
            or before.get('kernel_sha256') != SOURCE_KERNEL_SHA256
            or len(TARGET_KERNEL_SHA256) != 64):
        raise ValueError('Extension height requires the pinned source and reviewed target kernels')
    extensions, changed = before.get('extensions'), after.get('extensions')
    if (not isinstance(extensions, dict) or not isinstance(changed, dict)
            or extensions.get('jump_extension_height_floor', False) is not False
            or extensions.get('jump_peak_completion') is not True
            or extensions.get('jump_physical_tail') is not True
            or type(extensions.get('stationary_quiet')) not in (int, float)
            or extensions.get('stationary_quiet') != 1.
            or changed.get('jump_extension_height_floor') is not True):
        raise ValueError('Extension height permits only the fixed opt-in on the settled source')
    expected = deepcopy(source)
    expected['reward_kernel'] = {
        **before, 'kernel_sha256': TARGET_KERNEL_SHA256,
        'extensions': {**extensions, 'jump_extension_height_floor': True},
    }
    if target != expected:
        raise ValueError('Extension height cannot change other weights or physical/task/source identity')
    return {'format': FORMAT, 'before': deepcopy(source), 'after': deepcopy(target)}


def verify_saved_proof(proof):
    """Retain exact historical kernel endpoints under future running kernels."""
    if not isinstance(proof, dict) or proof.get('format') != FORMAT:
        raise ValueError('Unknown extension height proof')
    try:
        checked = verify_jump_extension_height_floor(proof['before'], proof['after'])
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete extension height proof') from error
    if checked != proof:
        raise ValueError('Extension height proof contains unregistered fields or source changes')
    return checked


def upgrade_jump_extension_height_floor(saved, dataset, target_schema):
    """Append one authenticated source change; preserve every learning/data object."""
    return _upgrade_reward_pair(saved, dataset, target_schema,
        target_kernel=TARGET_KERNEL_SHA256, verify=verify_jump_extension_height_floor,
        proof_key='jump_extension_height_floor')
