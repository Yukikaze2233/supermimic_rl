"""Caller height targets enter the same native and imagined command admission."""
import torch


def validate_height_override(target, height_override, height_override_mask):
    """Check tensor metadata only; existing admission owns command validity."""
    if height_override is None and height_override_mask is None:
        return
    if (not isinstance(height_override, torch.Tensor)
            or not isinstance(height_override_mask, torch.Tensor)
            or height_override.shape != target.shape[:-1]
            or height_override_mask.shape != target.shape[:-1]
            or height_override.dtype != target.dtype or height_override_mask.dtype != torch.bool
            or height_override.device != target.device or height_override_mask.device != target.device):
        raise ValueError('Height override requires matching height values and one bool mask per command row')


def override_command_height(target, height_override=None, height_override_mask=None):
    """Replace requested height only, without mutating or admitting the target."""
    validate_height_override(target, height_override, height_override_mask)
    if height_override is None:
        return target
    height = torch.where(height_override_mask, height_override, target[..., 3])
    return torch.cat((target[..., :3], height[..., None]), -1)
