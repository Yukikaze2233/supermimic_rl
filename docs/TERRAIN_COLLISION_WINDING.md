# Procedural step collision winding

The pinned native generator's `_platforms` implementation creates upward
platforms but winds positive-direction risers toward the higher platform. A
triangle mesh has an oriented surface: the riser must face the lower platform,
and adjacent faces must traverse their shared edge in opposite directions.
This also matters at the top edge, where contact generation combines adjacent
features. PhysX defines normals by `(b-a) cross (c-a)` ([SDK definition](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/struct_px_mesh_flag.html)).

`scripts/repair_terrain_winding.py` builds a separate derived atlas. It welds
exact-coordinate seams for the topology audit, seeds orientation from upward
platforms, and reverses inconsistent vertical triangles. It does not weld the
written collision mesh. The source atlas, coordinates, heights, materials,
triangle IDs and authored RMUC source meshes remain unchanged. Rebuilding the
query index must give exactly the previous arrays. Disconnected or conflicting
geometry is reported instead of guessing its inside/outside direction.

Run with the pinned native source on `PYTHONPATH`:

```sh
python scripts/repair_terrain_winding.py \
  --source assets/rmuc2027_course_atlas \
  --output outputs/terrain-winding/atlas \
  --report outputs/terrain-winding/report.json
```

The output must be new and outside the source directory. The report records
the source/candidate manifest hashes and every changed face. The checked atlas
with source hash `883dbed3068671a37f802dcae3e06c0a9ac136b8e9f7d0801712319b3494d6e0`
has 160 affected faces across 40 generated tiles. Tile 58 is a 147.998 mm
generated step, not either authored RMUC200mm course.

Geometry tests cover ascent, descent, reverse travel, duplicated seam vertices,
ambiguous topology, unchanged authored geometry and ground queries. Native
contact verification is separate: compare the same actor and initial setup on
the source/candidate atlas, inspect raw contact patches and body motion, and
retain the evaluation receipts. A correct mesh alone does not establish stair
traversal or jump capability.

A derived atlas has a new physical source identity. Do not overwrite a running
atlas or relabel an old replay as newly collected data. Any training handoff
must update the atlas, runtime and routing pins together, retain complete
optimizer/checkpoint state, and explicitly account for old samples from the
changed terrain. The repair tool neither changes training nor filters samples.

`world_model_v2.migrations.terrain.upgrade_terrain` performs that explicit
checkpoint conversion. It authenticates both runtimes/manifests and compares
the actual geometry arrays, allowing only reversed vertical triangles. It
retains all learned parameters, optimizer moments, normalization, random
streams and task clocks. Historical physical labels remain present; the new
replay's eligibility excludes intervals routed to a changed tile. The source
pair remains the record of their original eligibility. This is a physical
source change, not a force threshold or a success filter.

Auxiliary data retains its original file/hash and train/validation partition.
Only the registered source-to-target terrain/runtime mapping is accepted,
composed with an existing verified zero-disturbance mapping when necessary.
Changed-tile intervals are likewise excluded in memory; unchanged flat-flight
data and its sampler RNG remain available. Future online records on the fixed
terrain follow the ordinary validity rules without this historical exclusion.
