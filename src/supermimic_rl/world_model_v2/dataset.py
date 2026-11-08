"""Save native collection or evaluation in the existing compact V2 format."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from ..runtime import atomic_save
from .records import MODEL_CONTRACT


def save_native_dataset(collector, output: Path, collection: dict, *,
                        source_rng=(), stair_source_config=None, source=None) -> dict:
    """Preserve physics endpoints and task clocks, including unsuccessful attempts.

    This serializes the collector's records; it does not reconstruct them from
    observations or infer successful skills from the source policy's identity.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new native dataset path')
    if collector.replay is None or not len(collector.replay):
        raise ValueError('A native dataset requires collected transitions')
    env = collector.environment
    replay = collector.replay.state_dict()
    payload = {'format': 'supermimic-world-v2-dataset-v1', 'model_contract': MODEL_CONTRACT,
               'schema': env.schema.to_dict(), 'schema_fingerprint': env.schema.fingerprint,
               'replay': replay, 'collection': collection}
    sidecars = {}
    if source is not None:
        payload['source'] = source
    if collector.record_stair_source:
        from .stair_source import StairScenario

        trigger = env.adapter.env._terrain_course.tof_trigger
        config = (StairScenario.capture(env.adapter.env).to_dict()
                  if stair_source_config is None else stair_source_config)
        sequence = torch.tensor([entry[0] for entry in source_rng], dtype=torch.int64)
        if not torch.equal(sequence, replay['sequence']):
            raise ValueError('Native source RNG must match every recorded vector step')
        payload['stair_source_config'] = config
        payload['source_rng'] = {
            'format': 'supermimic-vector-step-rng-v1', 'device': str(trigger.generator.device),
            'num_envs': env.num_envs, 'sequence': sequence,
            'state': torch.stack([entry[1].cpu() for entry in source_rng]),
        }
        sidecars['stair-source-config.json'] = config
    if collector.record_jumps:
        from .jump_task import JumpTaskConfig

        payload['jump_task_config'] = asdict(JumpTaskConfig.capture(env.adapter.env))
        sidecars['jump-task-config.json'] = payload['jump_task_config']
    if collector.record_recovery:
        from .recovery_task import RecoveryTaskConfig

        payload['recovery_task_config'] = asdict(RecoveryTaskConfig.capture(env.adapter.env))
        sidecars['recovery-task-config.json'] = payload['recovery_task_config']
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, output)
    for name, value in sidecars.items():
        (output.parent / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(output), 'sha256': digest, 'bytes': output.stat().st_size,
            'model_contract': MODEL_CONTRACT, 'schema_fingerprint': env.schema.fingerprint,
            'vector_steps': len(replay['sequence']), 'transitions': len(collector.replay),
            'scope': 'Recorded native transitions; no skill qualification or learner update'}
