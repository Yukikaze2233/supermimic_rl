# Selected-platform flight target cost

The optional selected-platform task adds `cost.jump_target` using the existing
`jump_missed` coefficient. There is no additional floating-point weight. This
candidate is not evidence that a policy can land on a platform.

For each wheel, the source collision cylinder gives its centre `c`, radius `r`,
axis `a`, and half-length `l`. Its bottom is
`b = c.z - r * norm(a.xy) - l * abs(a.z)`. `PlatformGeometry.nearest` supplies
`p`, the XY-nearest point on the actual source top-face union with its actual Z.
It is not a contact patch or a landing certificate.

```
L = max(admitted_request_rise, r)
v = relu(p.z - b - existing_support_penetration_allowance)
u = (squared_norm(c.xy - p.xy) + v*v) / (L*L)
jump_target = jump_missed * mean_over_two_wheels(u / (1 + u))
```

The 5 ms integrator applies dt exactly once. A wheel above the platform has no
vertical target penalty. A low admitted rise still uses at least the real
wheel radius as its distance scale. No distance threshold is an admission or
termination rule.

Only the same still-live request's **after-observe** `FLIGHT` and
`TARGET_SINGLE_WHEEL` phases receive this cost. Each of the four physics ticks
has its own phase. `TARGET_BOTH_WHEELS` and subsequent continuation receive no
pull back toward the short platform. Before flight, wrong/ended requests,
missed/unknown contacts, and ordinary flat/STAIR tasks keep their existing
objectives. Missing or occluded geometry is reported unavailable and receives the existing
per-wheel upper bound of one. A known partner wheel retains its gradient. This
prevents unknown space from becoming cheaper than a visible target, without
certifying support or adding a simulation termination.

The original peak-missed cost remains unchanged. The sum of the peak and target
costs are bounded by `2 * jump_missed` (reachable for unavailable geometry). Keeping separate components preserves both
gradients: multiplying by peak completion or taking their maximum could remove
lateral learning while peak height is low. No positive tracking reward, jump
reference, flight/lift/push term, success qualification, or Actor input changes.

The native producer reuses this physics tick's cached actual wheel body pose
and source-cylinder transform. Reward does not perform another SDK pose read
or whole-cylinder clearance query. The world model uses its existing predicted
tree geometry with explicit passive knees. Both call the same distance formula;
whole-cylinder clearance remains a separate, nondifferentiated qualification
input. Identical explicit wheel poses have identical formula results; imperfect
predicted geometry need not equal actual native body geometry.

Future recorded contact/landing truth is not used by rollout. Current caller
intent, request binding, and the four phases advanced from predicted evidence
are explicit inputs. The selected-platform source receipt includes this
formula's version and implementation SHA. Old replay rewards are not rewritten.

Tests cover local finite differences, positive lateral/upward derivatives,
absence of a downward derivative above the platform, low-peak independent
lateral gradients, exact union holes and unavailable labels, actual native pose
reuse with nonzero world origin, and native/WM four-tick FLIGHT → SINGLE → BOTH
parity. These checks do not establish real-world derivatives or landing success.
