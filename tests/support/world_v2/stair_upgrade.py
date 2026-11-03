"""An online checkpoint upgraded to the stair-source contract."""

from dataclasses import asdict
import hashlib
import json
from supermimic_rl.contracts import StateSchema
from supermimic_rl.world_model_v2.migrations.motion import upgrade_motion_tasks
from supermimic_rl.world_model_v2.online import OnlineConfig
from supermimic_rl.world_model_v2.records import MODEL_CONTRACT
from supermimic_rl.world_model_v2.stair_source import StairScenario
from supermimic_rl.world_model_v2.stair_task import StairTaskConfig
from transformer_rl.formal.lab_env.stair_trigger import TofTriggerConfig
from support.world_v2.motion_upgrade import fixture as foundation_fixture
from support.world_v2.stair_task import native_stairs


def fixture():
    saved, data, _, _, _, _, boundary = foundation_fixture()
    saved, data, _ = upgrade_motion_tasks(saved, data)
    saved['task_norm_pending'] = False
    saved['config'] = asdict(OnlineConfig(cycles=64, record_tasks=True))
    atlas = json.dumps({'tiles': [{'kind': 0}]}).encode()
    scenario = StairScenario(StairTaskConfig.capture(native_stairs(2)), TofTriggerConfig(), .04, .20).to_dict()
    old_runtime = json.dumps({'stair_height_assist': scenario['task']['height_assist']}).encode()
    scenario['task']['height_assist']['enabled'] = True
    new_runtime = json.dumps({'stair_height_assist': scenario['task']['height_assist']}).encode()
    original = saved['schema']
    pins = {'terrain_sha256': hashlib.sha256(atlas).hexdigest(), 'asset': 'fixture'}
    schemas = [StateSchema.from_dict({**original, 'source': json.dumps({**pins,
               'runtime_sha256': hashlib.sha256(raw).hexdigest()})}) for raw in (old_runtime, new_runtime)]
    saved.update(schema=schemas[0].to_dict(), schema_fingerprint=schemas[0].fingerprint,
                 environment={'curriculum': {'real_transitions': 0, 'config': {'stage_transition_budgets': [10] * 8}}})
    data['schema_fingerprint'] = schemas[0].fingerprint
    reference = {'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
        'schema': schemas[1].to_dict(), 'schema_fingerprint': schemas[1].fingerprint,
        'stair_source_config': scenario}
    options = {'course_coordinate': 30, 'world_only_cycles': 4,
               'source_runtime_raw': old_runtime, 'target_runtime_raw': new_runtime}
    return saved, data, reference, atlas, options, boundary
