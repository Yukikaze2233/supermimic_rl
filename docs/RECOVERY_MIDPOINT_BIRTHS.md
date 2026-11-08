# Recovery midpoint birth candidate

This opt-in curriculum candidate addresses a gap in the PPO24100 fixed birth
evaluation: 52/88 first births passed; the existing side 45/60/75° and pitch 45°
bridges passed, but nine difficult endpoints still scored 0/36. The midpoint
angles interpolate those evaluated endpoints. They do not claim new recovery
capability, mechanical limits, or an explanation for every difficult failure.

Set `recovery_midpoint_bridge: true` in the existing course configuration,
alongside `recovery_roll_bridge: true`. Both the native 16 posture IDs and old
six bridge IDs stay unchanged. The flag defaults to false and is omitted from
the serialized course when false; the old 22-label distribution, source receipt
and evaluation selection remain unchanged.

| New variant ID | Label | Native base ID | Requested roll / pitch | Existing endpoints |
| --- | --- | --- | --- | --- |
| 22 | `left_side_82p5` | 4 | −82.5° / 0° | −75° / −90° |
| 23 | `right_side_82p5` | 5 | +82.5° / 0° | +75° / +90° |
| 24 | `pitch_forward_67p5` | 8 | 0° / +67.5° | +45° / +90° |
| 25 | `pitch_backward_67p5` | 9 | 0° / −67.5° | −45° / −90° |

The persisted recovery-family birth ordinal selects three full 16-pose cycles:

1. All original 16 postures.
2. Original easy slots 0, 1, 2, 3, 6, 7 become the six old side bridges.
3. Easy slots 0, 1, 2, 3 become the four midpoints; slots 6 and 7 retain the
   original forward/backward pitch-45 births.

Every 48 recovery births therefore contain 30 original difficult, eight easy,
six old bridge and four midpoint births. Original difficult share stays 62.5%;
each original difficult endpoint occurs three times. The nine endpoints that
failed the cited evaluation retain 27/48 = 56.25% of recovery births. Old bridge
share decreases from 18.75% to 12.5%, easy share from 18.75% to 16.67%, and
midpoints gain 8.33%. RECOVERY's share among task families does not change.

The rule activates only after the complete native 16-pose pool is available.
It consumes no additional RNG. Curriculum `state_dict` restores the ordinal;
the existing PPO cold physics resume does **not** restore the native sampler.
The new receipt retains that distinction. Old world-model releases and their
datasets are not upgraded by enabling this candidate.

`RecoveryPoseCompiler` produces each requested pose from its native base and
the roll/pitch offset, then `AssetCollisionGeometry.place` resolves height from
the actual compiled joints and collision geometry. Four additional static
zero-yaw local-flat placements are cached at initialization; arbitrary reset
orientations are not sampled. Nonzero yaw/elevated origins retain the existing
placement path. Native collision placement currently covers the 19 core bodies;
its `geometry_ready` result is not certification of the two independently
controlled gimbal bodies or of dynamic recovery success.

Birth logs distinguish requested roll/pitch, placed root pose, and first native
snapshot roll/pitch. A later native birth validation must also inspect the first
policy-owned pose after the unchanged preparation/release sequence; the first
snapshot is not that observation. No actor input, controller, action limit,
reward, recovery lifecycle, success qualification or gimbal command is changed.

For a native validation of only these four births, use the existing fixed
evaluation entry point with:

```json
{
  "family": "RECOVERY",
  "recovery_midpoints": true,
  "scenarios": [
    "recovery_left_side_82p5",
    "recovery_right_side_82p5",
    "recovery_pitch_forward_67p5",
    "recovery_pitch_backward_67p5"
  ]
}
```

`recovery_bridges: true` still selects the original 22 cases unless midpoints
are explicitly requested. Fixed rows retain their authored variants across
resets; variant IDs never change the native success predicates. Gimbal pitch
targets balance repetitions of each midpoint using the existing endpoints.

CPU tests cover default compatibility, three-cycle shares, irregular partial
resets and sampler restore, unchanged RNG/non-recovery rows, native placement
and cached pose isolation, invalid inputs, distinct first-snapshot logging and
explicit fixed evaluation lifecycle. These tests do not substitute for a native
dynamic birth validation or a new independent capability evaluation.
