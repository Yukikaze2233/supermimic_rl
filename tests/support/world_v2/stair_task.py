"""Native stair-request protocol state and completed estimates."""

from types import SimpleNamespace
import torch
from transformer_rl.formal.lab_env.course_tasks import StairRequestConfig, StairRequestProtocol
from transformer_rl.formal.lab_env.skill_fsm import StairFSM, StairFSMConfig, StairHeightAssistConfig
from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestProtocolConfig


def native_stairs(n, *, labels=False, assist=False):
    """Synthetic request timelines, not sensor/robot parameter recommendations."""
    zeros = lambda: torch.zeros(n)
    ids = lambda: torch.full((n,), -1, dtype=torch.int64)
    native = SimpleNamespace(num_envs=n, _device=torch.device('cpu'), step_dt=.02,
        time=torch.zeros(n, dtype=torch.float64), _command_base=torch.zeros(n, 4),
        _command=torch.zeros(n, 4), _base_command_mode=ids(), _command_mode=ids(),
        _ground_at_root=zeros(), _request_height=zeros(), _completed_estimate=None,
        cfg=SimpleNamespace(terrain_policy_spec=SimpleNamespace(use_mode_labels=labels),
            stair_height_assist_spec=StairHeightAssistConfig(assist, .07, .40, .06)),
        _skill_fsm=SimpleNamespace(stair=StairFSM(n, StairFSMConfig(policy_dt_s=.02))))
    native._clock = lambda: native.time
    course = LabTerrainCourse.__new__(LabTerrainCourse)
    native._terrain_course, course.env = course, native
    course.stair_requests = StairRequestProtocol(n, config=StairRequestConfig(policy_dt_s=.02, timeout_s=.20))
    course.stair_requests.reset(torch.arange(n), native.time)
    course.tasks = SimpleNamespace(active_id=ids(), protocol=SimpleNamespace(config=RequestProtocolConfig(policy_dt_s=.02)))
    course.tasks.step = lambda *args, **kwargs: (
        SimpleNamespace(admitted=native.admitted), native.base_mode, zeros(), None)
    course._pending_requests = {'jump': None, 'stair': None}
    course.manual_stair_request = torch.zeros(n, dtype=torch.bool)
    course.manual_stair_request_id, course.tof_request_id, course.stair_cancel_id = ids(), ids(), ids()
    course.manual_stair_height_m, course.tof_step_height_m = zeros(), zeros()
    course.stair_next_id = torch.zeros(n, dtype=torch.int64)
    course._last_tof_beam = None
    course.tof_trigger = SimpleNamespace(completion=lambda *a, **kw: ids(),
                                        record_responses=lambda *a, **kw: None, arm=lambda *a: None)
    course._tof_command = lambda request, height, *args, **kwargs: (request, height, torch.zeros(n, dtype=torch.bool))
    course._tof_rear_clearance = lambda beam: zeros()
    course._record_stair_end = lambda *args: None
    course._stair_ground_start, course._stair_request_height = zeros(), zeros()
    course._last_stair_response, course.stair_outcomes = ids(), torch.zeros(3, dtype=torch.int64)
    course._stair_course_contact_allowed = torch.arange(n) % 2 == 0
    course.body_contact_allowed = torch.zeros(n, dtype=torch.bool)
    return native


def estimate_at(timestamp, n, tilt=0.):
    return CompletedEstimate(torch.zeros(n), torch.full((n, 2), .9), torch.full((n,), tilt),
                             torch.zeros(n), timestamp.clone(), torch.ones(n, dtype=torch.bool))
