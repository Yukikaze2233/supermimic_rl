# Automatic jump admission and pending requests

`jump_takeoff.automatic_pending_admission=true` fixes two automatic-request
scheduling boundaries. It changes neither the physical readiness thresholds nor
the request protocol, reward, Actor inputs, action limits, or success criteria.

For example, a scheduled request sees estimated wheel contact probability .79
for one tick and .9 the next tick while physical readiness holds. Previously
the first `NOT_READY` consumed the ID and scheduled another opportunity 3–8 s
later. With this option, the intent remains pending for the next policy tick;
selection generates a **new ID**, respecting the original protocol's consumed-ID
semantics. Once accepted, the existing randomized interval is retained. The
native full-batch random vector is still drawn exactly as before; retry does not
add a draw or rewind a random generator.

Physical readiness now runs after this tick's command admission, including any
slew limit. If a segment changes its height target at the request boundary, the
gate and the native `supported_start` latch use the same admitted height. The
episode clock has already advanced at that point and is not advanced twice.
This avoids accepting a request whose supported-start qualification is already
false because the gate checked a previous height.

External packets retain their existing rejection/ID behavior; caller-owned rows
and owned empty packets never become automatic retries. Requests already in
progress are neither cancelled nor rescheduled. Step-local ownership/context is
cleared on normal return and exceptions. There is no new persistent queue,
record tensor, command lock, timeout, or threshold.

The option is disabled by default. Missing or explicit `false` retains the old
trigger receipt and serialized `JumpTaskConfig` field set as well as scheduling
behavior. An opt-in configuration carries `automatic_pending_admission=true`
and the `automatic_scheduling.version=admitted_height_pending_retry_v1` receipt.
The derived opt-in configuration type exists only to preserve the exact default
dataclass serialization contract.

The shared world-model `JumpTask.advance` implements the same admitted-height
gate and next-tick retry using its existing recorded task state and prepared
random draws. Native/imagined state equality is tested, including caller-owned
rows on the newer native entry-point source. This code parity does **not** migrate
old replays or make old online checkpoints claim the new native source. A future
WM deployment must explicitly adopt and verify the new source/configuration;
the existing online run is not changed by enabling this PPO option.

Existing cumulative native counters now appear at the original progress/log
boundary:

- `course/jump/request_response/<name>`: all request-protocol responses;
- `course/jump/external_request_response/<name>`: responses to caller packets;
- `course/jump/request_end/<name>`: live protocol end reasons;
- `course/jump/gate_deferred_row_ticks`: delayed environment-ticks, not distinct
  requests. One pending row can contribute repeatedly.

Physical completion counts remain separate. `NONE` includes ticks without a
request; `NOT_READY` is not a failed physical jump. These counters help distinguish
delayed opportunities from admitted requests and actual physical outcomes, but
do not by themselves identify every readiness failure condition.

The bugs were reproduced with constructed CPU states through the actual native
task path, including a simultaneous segment/request boundary. Such examples
prove the scheduling behavior; they do not establish its frequency or its share
of the low jump exposure observed in training. Ability improvements still need
normal continued training and independent native evaluation.
