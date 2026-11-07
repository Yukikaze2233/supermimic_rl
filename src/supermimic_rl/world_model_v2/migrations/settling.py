"""Authenticate the two reviewed settling changes without relabeling old data."""
from copy import deepcopy

from .reward_continuity import _upgrade_reward_pair


FORMAT = 'settling-reward-continuity-v1'
SOURCE_KERNEL_SHA256 = '8e86f622f8b408f1784e10da88e7fbb661210e0af70df363294052836d8c7838'
# Combined stationary quiet and physical-tail kernel from the reviewed changes.
# Historical proofs bind this file, not the implementation running in the future.
TARGET_KERNEL_SHA256 = '7ab9e5481d13cfc029852aa0ee44c0a3cf4cdb5672b3b269e8878cff4257850e'


def verify_settling_reward_continuity(source, target):
    """Allow only quiet=1 and physical-tail=True on the known peak-aware kernel."""
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('Settling reward proof requires complete source objects')
    before, after = source.get('reward_kernel'), target.get('reward_kernel')
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or before.get('implementation') != 'shared_physical_reward_densities'
            or before.get('kernel_sha256') != SOURCE_KERNEL_SHA256
            or len(TARGET_KERNEL_SHA256) != 64):
        raise ValueError('Settling reward requires the pinned source and reviewed target kernels')
    extensions, changed = before.get('extensions'), after.get('extensions')
    if (not isinstance(extensions, dict) or not isinstance(changed, dict)
            or extensions.get('jump_peak_completion') is not True
            or type(extensions.get('stationary_quiet', 0.)) not in (int, float)
            or extensions.get('stationary_quiet', 0.) != 0.
            or extensions.get('jump_physical_tail', False) is not False
            or type(changed.get('stationary_quiet')) not in (int, float)
            or changed.get('stationary_quiet') != 1.
            or changed.get('jump_physical_tail') is not True):
        raise ValueError('Settling reward permits only the fixed opt-ins on existing peak completion')
    expected = deepcopy(source)
    expected['reward_kernel'] = {
        **before, 'kernel_sha256': TARGET_KERNEL_SHA256,
        'extensions': {**extensions, 'stationary_quiet': 1., 'jump_physical_tail': True},
    }
    if target != expected:
        raise ValueError('Settling reward cannot change other weights or any physical/task/source identity')
    return {'format': FORMAT, 'before': deepcopy(source), 'after': deepcopy(target)}


def verify_saved_proof(proof):
    """Check the fixed historical kernel pair independently of the running file."""
    if not isinstance(proof, dict) or proof.get('format') != FORMAT:
        raise ValueError('Unknown settling reward proof')
    try:
        checked = verify_settling_reward_continuity(proof['before'], proof['after'])
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete settling reward proof') from error
    if proof != checked:
        raise ValueError('Settling reward proof contains unregistered fields or source changes')
    return checked


def upgrade_settling_reward_continuity(saved, dataset, target_schema):
    """Return a complete in-memory pair, preserving teacher, learners and labels.

    The native target reference and retained teacher/auxiliary restoration are
    checked by the handoff preflight. This function neither qualifies a teacher
    nor changes its source, writes a checkpoint, samples data or warms a learner.
    """
    return _upgrade_reward_pair(saved, dataset, target_schema,
        target_kernel=TARGET_KERNEL_SHA256, verify=verify_settling_reward_continuity,
        proof_key='settling_reward_continuity')
