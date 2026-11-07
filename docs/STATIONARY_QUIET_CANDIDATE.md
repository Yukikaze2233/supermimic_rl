# Supported stationary settling candidate

`RewardExtensions.stationary_quiet` defaults to `0`. The candidate adds a soft
settling objective after recovery hands control back to an explicit stationary
NORMAL command. It does not lock a state, limit an action, or change success.

For eligible physics ticks the additional positive density is

```text
weight * height_tracking_score * max(upright, 0)
       * exp(-sum(body_angular_velocity**2) / 0.25)
```

The height, upright factor and angular scale follow the existing recovery quiet
objective. `weight = 1.0` is the proposed first candidate because the current
recovery quiet density already uses that strength. This is a continuity
hypothesis, not evidence of improved native ability or an optimum weight.

Eligibility requires all of the following:

- NORMAL mode and exactly zero admitted forward, lateral and yaw command;
- no active special gimbal command;
- both wheel body loads at least the existing 5 N support threshold;
- no non-wheel load passing the existing harmful-contact predicate (`> 3 N`),
  with harmful-contact coverage available;
- no active terrain, terrain airborne or rough-landing reward window;
- no live physical jump request, including a NORMAL protocol-completed tail.

Missing explicit physical-jump activity raises an error when this extension is
enabled. An inactive request cannot be inferred from a zero command, a finished
reference stage or a retained requested jump height. The derived `physical_active`
field is read after each native physics observation, including success within
a policy interval. Imagination derives the same boolean from the existing jump
state and forwards it to the shared kernel. No replay column, Actor input,
network shape, request clock or native success rule changes.

The rough-landing exclusion also applies to flat-ground transient landing
windows. Existing recovery and jump rewards retain responsibility while their
own modes or physical jump tails are active. This candidate therefore does not
address an active jump's landing-tail objective by itself.

With weight zero, the extension is omitted from the native extension receipt
and old reward values require no new activity field. The kernel file digest
still changes; enabling this candidate on saved data requires a separately
authenticated source migration. This change does not provide that migration,
alter online weights, or restart training.

CPU verification covers mask boundaries, missing activity/coverage, old defaults,
native reset and within-interval success, native/imagined activity agreement,
the full imagined reward assembly, and finite-difference angular/height
gradients. These are implementation checks, not a robot capability evaluation.
