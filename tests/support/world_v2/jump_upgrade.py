"""An online checkpoint upgraded to the jump task contract."""

from dataclasses import asdict
import hashlib
import json
import torch
from supermimic_rl.contracts import StateSchema
from supermimic_rl.course.curriculum import CourseCurriculumConfig
from supermimic_rl.course.jump_first import make_course_curriculum
from supermimic_rl.world_model_v2.jump_task import JumpTaskConfig, JumpTaskState
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.migrations.stair import upgrade_stair_source
from support.world_v2.jump_task import native_jumps
from support.world_v2.stair_upgrade import fixture as stair_fixture


def fixture(*, change_course=True):
    saved, dataset, reference, atlas, options, boundary = stair_fixture()
    saved, dataset, _ = upgrade_stair_source(saved, dataset, reference, atlas, **options)
    saved['critic_norm_from'] = None
    saved['config']['actor_frame_dim'] = 35
    saved['collection_steps'] = 1234
    old_course = CourseCurriculumConfig(stage_transition_budgets=(10,) * 8).to_dict()
    current = make_course_curriculum(2, 'cpu', 19, old_course)
    current.set_progress(30, 768)
    current.sample(torch.arange(2))
    saved['environment']['curriculum'] = current.state_dict()
    target_course = ({'schedule': 'jump_first_v1', 'base': {
        **old_course, 'stage_transition_budgets': [20] * 8}} if change_course else old_course)
    source_raw, target_raw = [json.dumps(value).encode() for value in (old_course, target_course)]
    source = json.loads(saved['schema']['source'])
    source.update(course_sha256=hashlib.sha256(source_raw).hexdigest(), curriculum=old_course)
    target = {**source, 'course_sha256': hashlib.sha256(target_raw).hexdigest(), 'curriculum': target_course}
    if change_course:
        target['course_schedule'] = {'schedule': 'jump_first_v1', 'request_heading_override': False,
            'native_handoff_preserved': True, 'early_jump_domain': list(current.initial_domain.jump_m)}
    old_schema, target_schema = [StateSchema.from_dict({**saved['schema'], 'source': json.dumps(value)})
                                for value in (source, target)]
    saved.update(schema=old_schema.to_dict(), schema_fingerprint=old_schema.fingerprint)
    dataset['schema_fingerprint'] = old_schema.fingerprint
    native = native_jumps(2)
    records = {name: value[None] for name, value in JumpTaskState.capture(native).record().items()}
    records['jump_record_valid'] = torch.ones(1, 2, 1)
    reference = {'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
        'schema': target_schema.to_dict(), 'schema_fingerprint': target_schema.fingerprint,
        'stair_source_config': saved['stair_scenario'], 'jump_task_config': asdict(JumpTaskConfig.capture(native)),
        'replay': {'storage': records}}
    options = {'source_course_raw': source_raw, 'target_course_raw': target_raw,
               'course_coordinate': 45, 'world_only_cycles': 2}
    return saved, dataset, reference, options, boundary
