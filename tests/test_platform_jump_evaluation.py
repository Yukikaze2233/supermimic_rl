"""CPU mock environment ordering, using the real jump and stair protocols."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

UPSTREAM = Path(os.environ.get('SUPERMIMIC_UPSTREAM_ROOT',
    Path(__file__).resolve().parents[1] / '.sources/transformer_rl-41e8280'))
if (UPSTREAM / 'src').is_dir():
    sys.path.insert(0, str(UPSTREAM / 'src'))

native = pytest.importorskip('transformer_rl.formal.lab_env.course_tasks')
from transformer_rl.formal.tasks.curriculum import TaskFamily
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestProtocol, RequestProtocolConfig, RequestResponse
from supermimic_rl.course.platform_jump_evaluation import ManualPlatformJumpEvaluation, ManualPlatformJumpIntent
from supermimic_rl.course.jump_takeoff import JumpTakeoffConfig, JumpTriggerGate


class Stream:
    def __init__(self, n):
        self.requested = []
        self.n = n
        self.current = torch.tensor([[.5, 0., 0., .32]]).expand(n, -1).clone()
        self.height_cap = None

    def step(self, requested):
        self.requested.append(requested.clone())
        admitted = requested.clone()
        if self.height_cap is not None:
            admitted[:, 3].clamp_(max=self.height_cap)
        self.current.copy_(admitted)
        return SimpleNamespace(admitted=admitted)


class Course:
    def __init__(self, env):
        self.env = env
        n = env.num_envs
        protocol = RequestProtocol(n, config=RequestProtocolConfig(policy_dt_s=.02))
        self.tasks = SimpleNamespace(protocol=protocol, reference=native.JumpReferenceProgram(),
            stream=Stream(n), elapsed=torch.zeros(n), policy_dt_s=.02,
            next_id=torch.zeros(n, dtype=torch.long), family=torch.full((n,), int(TaskFamily.TERRAIN)),
            latched_height=torch.full((n,), .32), supported_start=torch.ones(n, dtype=torch.bool),
            _last_supported=torch.ones(n, dtype=torch.bool), _last_world_height=torch.full((n,), .32))
        self.stair_requests = native.StairRequestProtocol(n, config=native.StairRequestConfig(policy_dt_s=.02))
        self.router = SimpleNamespace(identity='atlas', tile=torch.full((n,), 101),
            manifest={'files': {'geometry.npz': {'sha256': 'geometry'}}})
        self._pending_requests = {'jump': None, 'stair': None}
        self.tof_calls = self.tof_countdown_ticks = 0
        self.tof_fire = False

    def submit_request(self, skill, packet):
        packet.validate(self.env.num_envs, self.env.device)
        assert self._pending_requests[skill] is None
        self._pending_requests[skill] = packet.owned()

    def _tof_command(self, request_id, height_m, cancel_id, safety_abort, ready, completed_id=None):
        self.tof_calls += 1
        self.tof_countdown_ticks += 1
        owned = (request_id >= 0) | (cancel_id >= 0) | safety_abort
        fire = (ready & ~owned & (self.tasks.protocol.active_id < 0)
                & (self.stair_requests.active_id < 0) & self.tof_fire)
        return (torch.where(fire, self.stair_requests.max_seen_id + 1, request_id),
                torch.where(fire, .2, height_m), fire)

    def advance_skill(self):
        t, e = self.tasks, self.env
        blocked = self.stair_requests.active_id >= 0
        packet = self._pending_requests['jump']
        self._pending_requests['jump'] = None
        t.elapsed += t.policy_dt_s
        requested = torch.tensor([[.5, 0., 0., .32]]).expand(e.num_envs, -1).clone()
        admission = t.stream.step(requested)
        ids = torch.full((e.num_envs,), -1, dtype=torch.long)
        estimate = CompletedEstimate(torch.zeros(e.num_envs), torch.ones(e.num_envs, 2),
            torch.zeros(e.num_envs), torch.zeros(e.num_envs), e._clock() - .02, e.ready.clone())
        jump = t.protocol.step(now_s=e._clock(), estimate=estimate,
            request_id=ids if packet is None else packet.request_id,
            requested_height_m=torch.zeros(e.num_envs) if packet is None else packet.height_m,
            cancel_id=ids, allow_new=(~blocked if packet is not None else torch.zeros_like(blocked)),
            safety_abort=torch.zeros_like(blocked))
        accepted = jump.response == int(RequestResponse.ACCEPTED)
        t.next_id = torch.maximum(t.next_id, t.protocol.max_seen_id + 1)
        t.latched_height = torch.where(accepted, admission.admitted[:, 3], t.latched_height)
        stair = self._pending_requests['stair']
        self._pending_requests['stair'] = None
        cancel = ids if stair is None else stair.cancel_id
        request, height, _ = self._tof_command(ids, torch.zeros(e.num_envs), cancel,
                                               torch.zeros_like(blocked), e.ready)
        condition = self.stair_requests.step(now_s=e._clock(), request_id=request, height_m=height,
            cancel_id=cancel, jump_active=t.protocol.active_id >= 0, ready=e.ready)
        e._skill_fsm.stair.state.copy_(condition.active.long())

    def reset_commands(self, ids):
        mask = torch.zeros(self.env.num_envs, dtype=torch.bool)
        mask[ids] = True
        self.tasks.protocol.reset(mask, self.env._clock())
        self.stair_requests.reset(ids, self.env._clock())
        self.tasks.elapsed[ids] = 0
        self.env._skill_fsm.stair.state[ids] = 0


class Adapter:
    def __init__(self, n=1):
        self.num_envs, self.device = n, torch.device('cpu')
        self._evaluation_configuration = {'family': 'TERRAIN'}
        self.time = 0.
        self.env = SimpleNamespace(num_envs=n, device=self.device, _episode_id=torch.zeros(n, dtype=torch.long),
            _clock=lambda: torch.full((n,), self.time, dtype=torch.float64), ready=torch.ones(n, dtype=torch.bool),
            _ground_at_root=torch.zeros(n),
            _skill_fsm=SimpleNamespace(stair=SimpleNamespace(state=torch.zeros(n, dtype=torch.long))))
        self.env._terrain_course = Course(self.env)
        # Test-only shortened episode/cooldown; native height tolerance is unchanged.
        self.env.supermimic_jump_trigger = JumpTriggerGate(self.env._terrain_course.tasks,
            JumpTakeoffConfig(min_episode_s=0., cooldown_s=0.))
        self.finished = False

    def evaluation_begin(self, schema):
        self._course_evaluator = SimpleNamespace(plan=SimpleNamespace(row_names=('stair',) * self.num_envs,
                                                           scenarios={'stair': {'family': 'TERRAIN'}}))
        self.reset(torch.arange(self.num_envs))
        self.evaluation_end = self.finish

    def evaluation_summary(self):
        return {'native_flat_success': 0}

    def evaluation_step(self, transition, tracking):
        return 'original_step'

    def reset(self, ids):
        self.env._terrain_course.reset_commands(ids)
        self.env._episode_id[ids] += 1  # Real native ordering: increment after reset_commands.

    def step(self):
        self.time = round(self.time + .02, 8)
        self.env._terrain_course.advance_skill()
        assert all(v is None for v in self.env._terrain_course._pending_requests.values())

    def finish(self):
        self.finished = True


@pytest.fixture
def setup(tmp_path):
    catalog = {'format': 'authored-platform-triangle-targets-v1',
               'source': {'atlas_manifest': {'sha256': 'atlas'}, 'atlas_geometry': {'sha256': 'geometry'}},
               'targets': [{'target_id': '101:profile:1', 'tile_id': 101, 'profile_index': 1,
                            'initial_landing_enabled': True}]}
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog))
    adapter = Adapter()

    def build(*, at=.02, rise=.2, height=.32):
        return ManualPlatformJumpEvaluation(adapter, intents=[ManualPlatformJumpIntent(0, '101:profile:1', at, rise, height)],
            catalog_path=path, catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sidecar_path=tmp_path / 'manual.jsonl')

    return adapter, build


def test_cancel_then_idle_then_fresh_jump_and_command_admission(setup):
    adapter, build = setup
    course = adapter.env._terrain_course
    with build() as manual:
        adapter.evaluation_begin(None)
        course.stair_requests.active_id[0] = 7
        course.stair_requests.max_seen_id[0] = 7
        adapter.env._skill_fsm.stair.state[0] = 1
        adapter.step()
        assert course.stair_requests.active_id[0] == -1
        assert not any(e['event'] == 'jump_packet' for e in manual.events)
        adapter.step()
        assert course.tasks.protocol.active_id[0] == 0
        assert course.tasks.latched_height[0] == pytest.approx(.32)
        assert course.tasks.stream.requested[-1][0, 3] == pytest.approx(.32)
        assert course.tasks.stream.requested[-1][0, :3].tolist() == [.5, 0., 0.]
        accepted = [e for e in manual.events if e['event'] == 'accepted_baseline'][0]
        assert accepted['admitted_support_height_m'] == pytest.approx(.32)
        assert course.tof_calls == course.tof_countdown_ticks == 2
        assert any(e['event'] == 'stair_cancel_response' and e['finished_id'] == 7 for e in manual.events)
        adapter.evaluation_end()
        report = adapter.evaluation_summary()
        assert report['native_flat_success'] == 0
        assert report['manual_platform_jump']['data_use']['joint_imagination_forbidden']
    assert adapter.finished
    assert 'advance_skill' not in course.__dict__
    assert 'step' not in course.tasks.protocol.__dict__


def test_not_ready_consumes_id_then_retry_is_fresh(setup):
    adapter, build = setup
    with build() as manual:
        adapter.evaluation_begin(None)
        adapter.env.ready[0] = False
        adapter.step()
        assert adapter.env._terrain_course.tasks.protocol.max_seen_id[0] == 0
        assert adapter.env._terrain_course.tasks.protocol.active_id[0] == -1
        adapter.env.ready[0] = True
        adapter.step()
        response = [e for e in manual.events if e['event'] == 'jump_response']
        assert [(e['request_id'], e['response_name']) for e in response] == [(0, 'not_ready'), (1, 'accepted')]
        assert manual.recording_source({'kind': 'test'})['manual_platform_jump']['data_use']['joint_imagination_forbidden']


def test_tof_detection_continues_and_window_releases_after_estimated_completion(setup):
    adapter, build = setup
    course = adapter.env._terrain_course
    with build() as manual:
        adapter.evaluation_begin(None)
        course.tof_fire = True
        adapter.step()
        assert manual.reserved[0] and course.stair_requests.active_id[0] == -1
        if course.tasks.protocol.active_id[0] < 0:
            adapter.step()  # The mock reset has not already primed a protocol boundary.
        assert course.tasks.protocol.active_id[0] >= 0
        # Force a real protocol estimated completion, not a privileged success.
        course.tasks.protocol.flight_seen[0] = True
        course.tasks.protocol.landing_since_s[0] = adapter.time - 2.
        adapter.step()
        assert manual.states[0]['stage'] == 'physical_tail'
        assert manual.reserved[0] and course.stair_requests.active_id[0] == -1
        for _ in range(310):
            adapter.step()
            if not manual.reserved[0]:
                break
        assert not manual.reserved[0]
        assert course.stair_requests.active_id[0] >= 0
        assert course.tof_countdown_ticks == course.tof_calls > 2
        assert any(e['event'] == 'intent_end' and e['reason'] == 'explicit_window_timeout' for e in manual.events)


def test_reset_separates_intents_and_preserves_native_id_watermark(setup):
    adapter, build = setup
    with build() as manual:
        adapter.evaluation_begin(None)
        adapter.step()
        adapter.reset(torch.tensor([0]))
        adapter.step()
        packets = [e for e in manual.events if e['event'] == 'jump_packet']
        assert [(e['episode_id'], e['request_id']) for e in packets] == [(1, 0), (2, 1)]
        assert sum(e['event'] == 'intent_end' and e['reason'] == 'native_reset' for e in manual.events) == 1
        transition = SimpleNamespace(start=SimpleNamespace(episode_id=torch.tensor([1])),
            terminated=torch.tensor([True]), truncated=torch.tensor([False]), valid=torch.tensor([False]))
        assert adapter.evaluation_step(transition, None) == 'original_step'
        event = manual.events[-1]
        assert event['episode_id'] == 1 and event['event'] == 'interval_end' and not event['valid']


def test_before_trigger_tof_unmodified_and_support_height_still_uses_stream(setup):
    adapter, build = setup
    with build(at=.04) as manual:
        adapter.evaluation_begin(None)
        adapter.env._terrain_course.tof_fire = True
        adapter.step()
        assert not manual.reserved[0]
        assert adapter.env._terrain_course.stair_requests.active_id[0] >= 0
        adapter.step()
        assert manual.reserved[0]
        assert adapter.env._terrain_course.stair_requests.active_id[0] == -1
        assert not any(e['event'] == 'jump_packet' for e in manual.events)
        adapter.step()
        assert any(e['event'] == 'jump_response' and e['response_name'] == 'accepted' for e in manual.events)


@pytest.mark.parametrize('field,value', [('jump_at_episode_s', -1), ('rise_m', float('nan')),
                                        ('support_height_m', 0), ('target_id', '102:profile:2')])
def test_intent_has_no_implicit_values(field, value):
    args = dict(env_id=0, target_id='101:profile:1', jump_at_episode_s=1., rise_m=.2, support_height_m=.32)
    with pytest.raises(ValueError):
        ManualPlatformJumpIntent(**{**args, field: value})


def test_wrong_fixed_tile_rejected_before_packet(setup):
    adapter, build = setup
    with build() as manual:
        adapter.evaluation_begin(None)
        adapter.env._terrain_course.router.tile[0] = 102
        with pytest.raises(ValueError, match='different target terrain'):
            adapter.step()
        assert not any(e['event'] == 'jump_packet' for e in manual.events)


def test_caller_height_must_be_admitted_before_native_readiness(setup):
    adapter, build = setup
    tasks = adapter.env._terrain_course.tasks
    tasks.stream.height_cap = .35  # Synthetic admission limit, never bypassed.
    with build(height=.4) as manual:
        adapter.evaluation_begin(None)
        adapter.step()
        adapter.step()
        assert tasks.stream.current[0, 3] == pytest.approx(.35)
        assert tasks.stream.requested[-1][0, 3] == pytest.approx(.4)
        assert not any(e['event'] == 'jump_packet' for e in manual.events)
        pending = [e for e in manual.events if e['event'] == 'pending']
        assert pending[-1]['reason'] == 'caller_height_not_yet_admitted'


def test_exact_dtype_height_and_native_supported_start_wait_then_issue(setup):
    adapter, build = setup
    tasks = adapter.env._terrain_course.tasks
    tasks._last_world_height[:] = .4
    with build(height=.32) as manual:
        adapter.evaluation_begin(None)
        adapter.step()
        assert not any(e['event'] == 'jump_packet' for e in manual.events)
        assert manual.events[-1]['reason'] == 'native_supported_start_not_ready'
        tasks._last_world_height[:] = .32
        adapter.step()
        packet = next(e for e in manual.events if e['event'] == 'jump_packet')
        assert packet['delay_after_earliest_s'] == pytest.approx(.02)
        assert tasks.protocol.active_id[0] == 0
        assert manual.summary()['trigger']['uses_privileged_native_supported_start']
        assert not manual.summary()['trigger']['deployment_controller_implemented']


def test_not_initiated_timeout_is_not_a_jump_failure(setup):
    adapter, build = setup
    tasks = adapter.env._terrain_course.tasks
    tasks._last_supported[:] = False
    with build() as manual:
        adapter.evaluation_begin(None)
        for _ in range(305):
            adapter.step()
        end = next(e for e in manual.events if e['event'] == 'intent_end')
        assert end['reason'] == 'explicit_window_timeout'
        assert end['initiation_outcome'] == 'not_initiated'
        assert not end['request_was_accepted']
        assert not any(e['event'] == 'jump_packet' for e in manual.events)
        assert not manual.reserved[0]
        assert tasks.protocol.max_seen_id[0] == -1
        assert manual.summary()['platform_landing_success'] is None


def test_exception_after_begin_restores_both_hook_layers(setup):
    adapter, build = setup
    course = adapter.env._terrain_course
    with pytest.raises(RuntimeError, match='collector setup'):
        with build():
            adapter.evaluation_begin(None)
            raise RuntimeError('collector setup failed before native evaluation try/finally')
    assert adapter.finished
    assert 'advance_skill' not in course.__dict__
    assert 'evaluation_begin' not in adapter.__dict__


def test_missing_existing_gate_is_not_replaced_by_new_thresholds(setup):
    adapter, build = setup
    del adapter.env.supermimic_jump_trigger
    with build():
        with pytest.raises(ValueError, match='existing native jump supported-start gate'):
            adapter.evaluation_begin(None)
    assert adapter.finished
