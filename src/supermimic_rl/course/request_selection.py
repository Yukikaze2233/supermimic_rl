"""Pure per-row selection between scheduled and caller-owned jump packets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand


@dataclass(frozen=True)
class JumpRequestSelection:
    request_id: Tensor
    requested_height_m: Tensor
    cancel_id: Tensor
    safety_abort: Tensor
    allow_new: Tensor
    next_id: Tensor
    next_request_s: Tensor
    automatic_due: Tensor
    packet_owned: Tensor


def validate_packet_owned(packet_owned, *, packet, count, device):
    """Validate ownership metadata without reading tensor values or sampling."""
    if packet_owned is not None and (packet is None or not isinstance(packet_owned, Tensor)
            or packet_owned.shape != (count,) or packet_owned.dtype != torch.bool
            or packet_owned.device != device):
        raise ValueError('packet_owned requires a packet and one bool per row on the request device')


def select_jump_request(*, task_family: Tensor, base_mode: Tensor,
                        blocked_by_stair: Tensor, elapsed_s: Tensor, active_id: Tensor,
                        next_id: Tensor, next_request_s: Tensor, rise_m: Tensor,
                        interval_uniform: Tensor | None = None,
                        packet: RequestCommand | None = None,
                        packet_owned: Tensor | None = None) -> JumpRequestSelection:
    """Return a packet and scheduler state without mutation or random sampling.

    Without an ownership mask, the original whole-batch contract applies:
    no packet means automatic scheduling and requires N uniform draws; a packet
    owns every row and requires no draw. An explicit mask uses N prepared draws
    even if every row happens to be owned, avoiding data-dependent RNG demand.
    Only unowned rows can consume an automatic ID or scheduling interval.

    Readiness, cancellation, ID watermarks and physical success remain in the
    existing trigger/protocol/evidence owners. This selector adds no admission
    condition, and an owned empty packet deliberately keeps that row dormant.
    """
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.tasks.curriculum import TaskFamily

    n, device = len(next_id), next_id.device
    validate_packet_owned(packet_owned, packet=packet, count=n, device=device)
    owned = (torch.full((n,), packet is not None, dtype=torch.bool, device=device)
             if packet_owned is None else packet_owned)
    if packet is not None:
        packet.validate(n, device)
    automatic = packet is None or packet_owned is not None
    request_family = task_family == int(TaskFamily.REQUEST)
    if automatic:
        if (interval_uniform is None or interval_uniform.shape != (n,)
                or interval_uniform.device != device or interval_uniform.dtype != torch.float32):
            raise ValueError('Automatic jump requests require one recorded/prepared uniform draw per row')
        due = (request_family & ~blocked_by_stair
               & (elapsed_s >= next_request_s) & (active_id < 0))
        if packet_owned is not None:
            due = due & ~packet_owned
        request_id = torch.where(due, next_id, -1)
        selected_next_id = next_id + due.long()
        selected_next_request = torch.where(due, elapsed_s + (3 + 5 * interval_uniform), next_request_s)
    else:
        due = torch.zeros(n, dtype=torch.bool, device=device)
        request_id = torch.full_like(next_id, -1)
        selected_next_id, selected_next_request = next_id, next_request_s
    height = rise_m
    cancel_id = torch.full_like(next_id, -1)
    safety_abort = torch.zeros(n, dtype=torch.bool, device=device)
    allowed = request_family
    if packet is not None:
        request_id = torch.where(owned, packet.request_id, request_id)
        height = torch.where(owned, packet.height_m, height)
        if packet.cancel_id is not None:
            cancel_id = torch.where(owned, packet.cancel_id, cancel_id)
        if packet.safety_abort is not None:
            safety_abort = owned & packet.safety_abort
        allowed = torch.where(owned, base_mode != int(CommandMode.RECOVERY), allowed)
    return JumpRequestSelection(request_id, height, cancel_id, safety_abort,
        allowed & ~blocked_by_stair, selected_next_id, selected_next_request, due, owned)
