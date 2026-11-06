"""Native collection and repeated world/BPTT updates with recoverable replay.

Fixed producers retain their selected coordinate. An explicit progression maps
collection steps to the pinned sampling schedule; passing through a stage does
not claim skill mastery. Teacher trajectories are data, not PPO updates or
behavior-cloning targets for the student.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time

import torch

from ..contracts import StateSchema
from ..chassis_observation import COMMAND_CONTRACTS, HEADING_COMMANDS, LEGACY_COMMANDS
from ..formal_observation import known_horizon
from ..runtime import atomic_save
from .auxiliary_data import AuxiliaryWorldData, WorldDatasetConfig
from .collection import Collector
from .course_progression import CourseProgression, CourseProgressionConfig
from .factory import make_learner
from .learning import LearningConfig
from .mixing import MixedDecision
from .motion_task import MotionTask, MOTION_TASK_DIM
from .records import MODEL_CONTRACT, RecordSpec, decode_physical, decode_memory
from .replay import CompactReplay
from .stair_source import StairScenario, StairSourceState
from .stair_task import StairTaskState
from .terrain import AtlasTerrain
from .jump_task import (JumpTaskConfig, JumpTaskState, JumpTask, JUMP_FLIGHT_LATCH_RECORD_FIELDS,
                        JUMP_RECORD_FIELDS, JUMP_TRIGGER_RECORD_FIELDS)
from .recovery_task import RecoveryTaskConfig, RecoveryTask, RECOVERY_RECORD_FIELDS
from .guard import NativeRegressionGuard, validate_guard_settings
from .validation import calibrate_contact_floor, validate

CHECKPOINT_FORMAT = 'supermimic-online-v2'


GUARD_SETTINGS = ('guard_drop_fraction', 'guard_patience', 'guard_window', 'guard_mode',
                  'guard_min_rate_fraction', 'guard_recover_after')


@dataclass(frozen=True)
class OnlineConfig:
    cycles: int = 128
    collection_steps: int = 24
    replay_steps: int = 1000
    replay_device: str = 'learner'
    model_updates: int = 96
    model_batch_size: int = 2048
    fit_horizon: int = 8
    actor_updates: int = 24
    imagination_batch_size: int = 4096
    imagination_microbatch_size: int = 512
    estimator_warmup_updates: int = 200
    action_noise_std: float = .1
    teacher_fraction: float = .5
    teacher_hold_updates: int = 800
    teacher_fade_updates: int = 800
    validation_fraction: float = .125
    validation_horizon: int = 32
    validation_batch_size: int = 512
    checkpoint_interval: int = 10
    replay_checkpoint_interval: int = 32
    curriculum_transition_cap: int = 0
    record_tasks: bool = False
    record_stair_source: bool = False
    record_jumps: bool = False
    record_recovery: bool = False
    course_coordinate: int | None = None
    course_progression: CourseProgressionConfig | None = None
    policy_start_cycle: int = 0
    seed: int = 7
    model_hidden: tuple[int, ...] = (2048, 2048, 2048)
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    actor_frame_dim: int = 45
    chassis_command_contract: str = LEGACY_COMMANDS
    actor_features: dict | None = None
    estimator_hidden: tuple[int, ...] = (128, 64)
    critic_hidden: tuple[int, ...] = (256, 128)
    learning: LearningConfig = field(default_factory=LearningConfig)
    world_datasets: tuple[WorldDatasetConfig, ...] = ()
    # A fresh run may copy a PPO actor (same 5x35 interface and bounds) into the student.
    initial_actor: str | None = None
    initial_actor_sha256: str | None = None
    # Warmup cycles fit the critic on the frozen actor's imagined returns
    # (instead of only the estimator) so the first actor step has a trained bootstrap.
    critic_warmup: bool = False
    # Floor non-wheel contact-load noise in imagination, recalibrated on held-out rows.
    calibrate_contact_floor: bool = False
    # Native regression guard: when the student's own native reward stays more
    # than this fraction below its best for guard_patience consecutive cycles,
    # restore the best actor and its optimizer state and halve the actor
    # learning rate. Imagined objectives cannot detect model exploitation;
    # native returns can. 0 disables it.
    guard_drop_fraction: float = 0.
    guard_patience: int = 2
    # Collections averaged per judgment, and "restore" or review-only "report" (see guard.py).
    guard_window: int = 1
    guard_mode: str = 'restore'
    # Restore mode: lowest actor rate as a base fraction, and good judgments that undo one halving.
    guard_min_rate_fraction: float = 0.
    guard_recover_after: int = 0

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ('model_hidden', 'actor_hidden', 'estimator_hidden', 'critic_hidden'):
            if key in value:
                value[key] = tuple(value[key])
        value['learning'] = LearningConfig(**value.get('learning', {}))
        value['world_datasets'] = tuple(WorldDatasetConfig(**item) for item in value.get('world_datasets', ()))
        if value.get('course_progression') is not None:
            value['course_progression'] = CourseProgressionConfig(**value['course_progression'])
        return cls(**value)

    def __post_init__(self):
        from ..actor_features import actor_feature_config
        sources = tuple(self.world_datasets)
        if (any(not isinstance(source, WorldDatasetConfig) for source in sources)
                or len({source.name for source in sources}) != len(sources)
                or sum(source.fraction for source in sources) >= 1):
            raise ValueError('Auxiliary world sources need distinct names and total fraction below one')
        object.__setattr__(self, 'world_datasets', sources)
        actor_feature_config(self.actor_features)
        if self.actor_features is not None and (self.actor_frame_dim != 35 or not self.record_jumps):
            raise ValueError('Actor reference features require the 35D actor and recorded jump tasks')
        if self.replay_device not in ('learner', 'cpu'):
            raise ValueError('Replay device must be learner or cpu')
        if type(self.actor_frame_dim) is not int or self.actor_frame_dim not in (35, 45):
            raise ValueError('Actor frame dimension must be 35 or 45')
        if (self.chassis_command_contract not in COMMAND_CONTRACTS
                or self.actor_frame_dim != 35 and self.chassis_command_contract != LEGACY_COMMANDS):
            raise ValueError('Projected chassis commands require the 35D actor')
        for name in ('cycles', 'collection_steps', 'replay_steps', 'model_updates', 'model_batch_size',
                     'fit_horizon', 'actor_updates', 'imagination_batch_size', 'imagination_microbatch_size',
                     'checkpoint_interval', 'replay_checkpoint_interval', 'validation_horizon', 'validation_batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer online setting required: ' + name)
        if (self.replay_steps < self.fit_horizon or not 0 < self.validation_fraction < 1
                or self.action_noise_std < 0 or self.estimator_warmup_updates < 0):
            raise ValueError('Invalid online replay, validation, noise or warmup setting')
        if self.curriculum_transition_cap != 0:
            raise ValueError('Foundation reward producer requires the initial course coordinate')
        if type(self.policy_start_cycle) is not int or self.policy_start_cycle < 0:
            raise ValueError('Policy start cycle must be a nonnegative integer')
        if self.record_stair_source:
            if not self.record_tasks or type(self.course_coordinate) is not int or self.course_coordinate < 0:
                raise ValueError('Stair collection requires command records and an explicit fixed course coordinate')
        elif self.course_coordinate is not None or self.policy_start_cycle:
            raise ValueError('Nonfoundation course placement/warmup requires the terrain and stair task producer')
        if self.record_jumps and not self.record_stair_source:
            raise ValueError('Jump collection requires the complete terrain and stair arbitration producer')
        if self.record_recovery and not self.record_jumps:
            raise ValueError('Recovery collection requires the complete jump and stair arbitration producer')
        if self.course_progression is not None:
            if (not isinstance(self.course_progression, CourseProgressionConfig) or not self.record_recovery
                    or self.course_coordinate != self.course_progression.start_coordinate):
                raise ValueError('Course progression requires the full task producer and matching initial coordinate')
        if (self.initial_actor is None) != (self.initial_actor_sha256 is None):
            raise ValueError('initial_actor and initial_actor_sha256 must be given together')
        validate_guard_settings(drop_fraction=self.guard_drop_fraction, patience=self.guard_patience,
                                window=self.guard_window, mode=self.guard_mode,
                                min_rate_fraction=self.guard_min_rate_fraction,
                                recover_after=self.guard_recover_after)
        if self.learning.anchor_coefficient and self.initial_actor is None:
            raise ValueError('The teacher anchor needs the PPO initial_actor it is centered on')
        if self.initial_actor is not None and self.actor_frame_dim != 35:
            raise ValueError('A PPO actor initializes only the 5x35 chassis student')


def policy_start_candidates(replay, config, env_ids, policy_dt_s):
    """Shared native-record eligibility for training and isolated BPTT audits."""
    key = ('online_starts', config.record_stair_source, config.record_tasks, config.record_jumps, config.record_recovery,
           tuple(env_ids), policy_dt_s)
    if key not in replay._cache:
        candidates = replay.start_candidates(env_ids)
        offset, env = candidates // replay.num_envs, candidates % replay.num_envs
        slots = (offset + replay.head - replay.size) % replay.capacity
        data = replay.data
        horizon, goal, route = (data[name][slots, env] for name in ('horizon', 'goal', 'reward_route'))
        if config.record_stair_source:
            allowed_mode = ((route[:, 0] <= 2) | ((route[:, 0] == 4) & config.record_jumps)
                            | ((route[:, 0] == 3) & config.record_recovery))
            family_limit = 5 if config.record_recovery else 4 if config.record_jumps else 3
            ordinary = (allowed_mode & ((goal[:, 7] <= .5) | config.record_recovery)
                        & (data['task_family'][slots, env, 0] <= family_limit)
                        & (data['task_record_valid'][slots, env, 0] > .5)
                        & (data['stair_source_valid'][slots, env, 0] > .5))
            if not config.record_jumps:
                ordinary &= goal[:, 8] <= .5
            jump = None
            if config.record_jumps:
                if any(name not in data for name in (*JUMP_RECORD_FIELDS, 'jump_record_valid')):
                    raise ValueError('Jump imagination requires complete native records; old rows are not inferred')
                ordinary &= data['jump_record_valid'][slots, env, 0] > .5
                jump = JumpTaskState.decode({name: data[name][slots, env] for name in
                                            (*JUMP_RECORD_FIELDS, *JUMP_TRIGGER_RECORD_FIELDS,
                                             *JUMP_FLIGHT_LATCH_RECORD_FIELDS) if name in data})
                if jump.trigger_flags is not None:
                    ordinary &= jump.trigger_flags[:, 0]
            if config.record_recovery:
                if any(name not in data for name in (*RECOVERY_RECORD_FIELDS, 'recovery_record_valid')):
                    raise ValueError('Recovery imagination requires native lifecycle records; old rows are not inferred')
                ordinary &= data['recovery_record_valid'][slots, env, 0] > .5
            recorded = {name: value[slots, env] for name, value in data.items()
                        if name.startswith(('stair_', 'tof_'))}
            known = StairSourceState.decode(recorded).boundary_known(
                MotionTask.decode(data['motion_task'][slots, env]), StairTaskState.decode(recorded), policy_dt_s, jump)
        elif config.record_tasks:
            ordinary = (route[:, :2] == 0).all(-1) & (goal[:, 7:9].amax(-1) <= .5)
            ordinary &= data['task_record_valid'][slots, env, 0] > .5
            dynamic = torch.zeros_like(horizon[:, 1])
        else:
            ordinary = (route == 0).all(-1) & (horizon[:, 1] == 0)
            dynamic = horizon[:, 1]
        if not config.record_stair_source:
            known = known_horizon(horizon[:, 0], dynamic, goal, policy_dt_s) >= 1
        if 'disturbance' in data:
            ordinary &= data['disturbance_record_valid'][slots, env, 0] > .5
        replay._cache[key] = candidates[(data['reward_context_valid'][slots, env, 0] > .5) & ordinary & known]
    return replay._cache[key]


class OnlineTrainer:
    def __init__(self, environment, teacher_decision, config: OnlineConfig, output: Path, *, runtime,
                 linkage_path, asset_directory, identified_bundle, seed_dataset=None, world_checkpoint=None,
                 resume=None, teacher_source=None):
        self.env, self.config, self.output = environment, config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device, self.cycle = environment.device, 0
        self.replay_device = torch.device(self.device if config.replay_device == 'learner' else 'cpu')
        torch.manual_seed(config.seed)
        self.generator = torch.Generator(device=self.replay_device).manual_seed(config.seed + 1)
        self.validation_generator = torch.Generator(device=self.replay_device).manual_seed(config.seed + 4)
        if environment.current is None:
            environment.reset()
        self.stair_scenario = None
        self.jump_scenario = None
        self.recovery_scenario = None
        terrain = stair_task = stair_source = jump_task = recovery_task = None
        if config.record_stair_source:
            scenario = StairScenario.capture(environment.adapter.env)
            self.stair_scenario = scenario.to_dict()
            terrain = AtlasTerrain.load(runtime['terrain_atlas_pin']['path'],
                expected_sha256=json.loads(environment.schema.source)['terrain_sha256'], device=self.device)
            stair_task, stair_source = scenario.build(terrain)
        if config.record_jumps:
            jump_config = JumpTaskConfig.capture(environment.adapter.env)
            self.jump_scenario = asdict(jump_config)
            jump_task = JumpTask(jump_config)
        if config.record_recovery:
            recovery_config = RecoveryTaskConfig.capture(environment.adapter.env)
            self.recovery_scenario = asdict(recovery_config)
            recovery_task = RecoveryTask(recovery_config)
        self.learner = make_learner(environment.schema, runtime, config.learning,
            linkage_path=linkage_path, asset_directory=asset_directory, identified_bundle=identified_bundle,
            device=self.device, model_hidden=config.model_hidden, actor_hidden=config.actor_hidden,
            estimator_hidden=config.estimator_hidden, critic_hidden=config.critic_hidden, record_tasks=config.record_tasks,
            terrain=terrain, stair_task=stair_task, stair_source=stair_source, jump_task=jump_task,
            recovery_task=recovery_task,
            actor_frame_dim=config.actor_frame_dim, chassis_command_contract=config.chassis_command_contract,
            actor_features=config.actor_features)
        self.teacher_source = teacher_source
        self.actor_initialization = None
        self.course_progression = None
        saved = None
        if resume is not None:
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            self._check_saved(saved)
            replay_path = Path(resume).parent / saved['replay_file']
            dataset = torch.load(replay_path, map_location='cpu', weights_only=True, mmap=True)
            if dataset['cycle'] != saved['cycle'] or dataset['schema_fingerprint'] != environment.schema.fingerprint:
                raise ValueError('Checkpoint and replay generation differ')
            self.learner.load_state_dict(saved['learner'])
            self._initialize_course_progression(saved)
            expected_coordinate = (config.course_coordinate if config.course_progression is None else
                                   config.course_progression.coordinate(max(0, saved['collection_steps'] - 1)))
            if config.record_stair_source and saved['environment']['curriculum']['real_transitions'] != expected_coordinate:
                raise ValueError('Resume course placement must match the explicitly migrated lookup coordinate')
            environment.load_training_state_dict(saved['environment'])
            self.cycle = saved['cycle']
            self.seed_source = saved['seed_source']
            self.actor_initialization = saved.get('actor_initialization')
        else:
            self._initialize_course_progression(None)
            if seed_dataset is None or world_checkpoint is None:
                raise ValueError('A fresh online run requires native seed data and its fitted world model')
            dataset = torch.load(seed_dataset, map_location='cpu', weights_only=True, mmap=True)
            if dataset.get('format') != 'supermimic-world-v2-dataset-v1' or dataset['model_contract'] != MODEL_CONTRACT:
                raise ValueError('Online seed data must use the current compact world-model contract')
            if (dataset['schema_fingerprint'] != environment.schema.fingerprint
                    or StateSchema.from_dict(dataset['schema']).fingerprint != environment.schema.fingerprint):
                raise ValueError('Seed data and live native environment have different contracts')
            if config.record_jumps and (dataset.get('jump_task_config') is None or
                    JumpTaskConfig.from_dict(dataset['jump_task_config']) != jump_config):
                raise ValueError('Seed jump task configuration differs from the native environment')
            if config.record_recovery and dataset.get('recovery_task_config') != self.recovery_scenario:
                raise ValueError('Seed recovery lifecycle configuration differs from the native environment')
            world = torch.load(world_checkpoint, map_location='cpu', weights_only=True)
            contract = world['model_contract']
            if (world.get('format') != 'supermimic-offline-world-v2' or contract['version'] != MODEL_CONTRACT
                    or contract['schema_fingerprint'] != environment.schema.fingerprint
                    or tuple(contract['hidden']) != config.model_hidden):
                raise ValueError('Initial world checkpoint does not match the live contract')
            self.learner.model.load_state_dict(world['model'])
            if config.initial_actor is not None:
                from .student_transfer import initialize_student_from_ppo
                self.actor_initialization = initialize_student_from_ppo(
                    self.learner.actor, Path(config.initial_actor), config.initial_actor_sha256)
                if config.learning.anchor_coefficient:
                    self.learner.set_anchor(self.learner.actor,
                                            torch.tensor(self.actor_initialization['action_std']))
            self.seed_source = {'dataset': str(Path(seed_dataset).resolve()),
                                'world_checkpoint': str(Path(world_checkpoint).resolve()),
                                'world_iteration': world['iteration']}
            if config.course_coordinate is not None:
                environment.set_training_progress(real_transitions=config.course_coordinate, actor_updates=0)
                environment.reset()
        raw = dataset['replay']
        if environment.num_envs > raw['num_envs']:
            raise ValueError('Seed data needs at least one distinct recorded row per live environment')
        rows = list(range(environment.num_envs)) if environment.num_envs != raw['num_envs'] else None
        replay = CompactReplay.from_state_dict(raw, self.replay_device,
                                              capacity_steps=config.replay_steps, env_ids=rows)
        expected = RecordSpec(environment.schema.goal_dim, 65).fields(record_tasks=config.record_tasks,
            record_terrain=config.record_stair_source, record_stairs=config.record_stair_source,
            record_stair_source=config.record_stair_source, record_jumps=config.record_jumps,
            record_recovery=config.record_recovery,
            record_jump_trigger=jump_task is not None and jump_task.config.trigger is not None,
            record_jump_flight_latch=jump_task is not None and jump_task.config.flight_latch,
            record_disturbances=self.learner.model.disturbances)
        if any(name not in replay.data or tuple(replay.data[name].shape[2:]) != shape for name, shape in expected.items()):
            raise ValueError('Online replay field layout differs from the current compact contract')
        self.mixing = MixedDecision(environment, self.learner.actor, teacher_decision, seed=config.seed + 2,
            initial_fraction=config.teacher_fraction, hold_updates=config.teacher_hold_updates,
            fade_updates=config.teacher_fade_updates)
        self.resume_migrations = list(saved.get('resume_migrations', [])) if saved is not None else []
        if getattr(self, 'actor_features_activated', False):
            self.resume_migrations.append({'actor_features': config.actor_features,
                'learner_replay_rng_preserved': True, 'additional_warmup_cycles': 0,
                'scope': 'Actor decoder input only; raw replay, estimator, dynamics and rewards preserved'})
        if getattr(self, 'chassis_commands_activated', False):
            self.resume_migrations.append({'chassis_command_contract': config.chassis_command_contract,
                'learner_replay_rng_preserved': True, 'additional_warmup_cycles': 0,
                'scope': 'project raw formal45 commands on read; dynamics, rewards and replay are unchanged'})
        if getattr(self, 'course_progression_activated', False):
            self.resume_migrations.append({'course_progression_activated': asdict(config.course_progression),
                'learner_replay_rng_preserved': True, 'additional_warmup_cycles': 0,
                'scope': 'sampling schedule only; skill progress remains independently evaluated'})
        if getattr(self, 'jump_height_progression_changed', None):
            self.resume_migrations.append({'jump_height_progression': self.jump_height_progression_changed,
                'learner_replay_rng_preserved': True, 'additional_warmup_cycles': 0,
                'scope': 'Future height sampling only; gate verdicts and course clocks preserved'})
        self.task_norm_pending = bool(saved.get('task_norm_pending', False)) if saved is not None else False
        self.critic_norm_from = saved.get('critic_norm_from') if saved is not None else None
        self.collector = Collector(environment, self.mixing, capacity_steps=config.replay_steps, replay=replay,
            seed=config.seed + 3, action_noise_std=config.action_noise_std,
            curriculum_transition_cap=config.curriculum_transition_cap, record_tasks=config.record_tasks,
            record_terrain=config.record_stair_source, record_stairs=config.record_stair_source,
            record_stair_source=config.record_stair_source, record_jumps=config.record_jumps,
            record_recovery=config.record_recovery, course_coordinate=config.course_coordinate,
            course_coordinate_at=None if config.course_progression is None else config.course_progression.coordinate)
        held = max(1, round(environment.num_envs * config.validation_fraction))
        if held >= environment.num_envs:
            raise ValueError('Online learning needs distinct training and validation environment rows')
        self.train_ids = list(range(environment.num_envs - held))
        self.validation_ids = list(range(environment.num_envs - held, environment.num_envs))
        self.auxiliary_world = (AuxiliaryWorldData(config.world_datasets, environment.schema, replay,
            config.fit_horizon, seed=config.seed + 5,
            disturbance_runtime_pairs=tuple(m['disturbance_runtime_pair'] for m in self.resume_migrations
                                            if 'disturbance_runtime_pair' in m)) if config.world_datasets else None)
        if self.auxiliary_world is not None and saved is not None and saved.get('auxiliary_world') is not None:
            self.auxiliary_world.load_state_dict(saved['auxiliary_world'])
        if getattr(self, 'world_datasets_changed', None) is not None:
            self.resume_migrations.append({'world_datasets': self.world_datasets_changed,
                'scope': 'Auxiliary dynamics windows only; live replay and all learner states preserved'})
        self._initialize_guard(saved)
        if saved is not None:
            self.collector.steps = saved['collection_steps']
            self.collector.generator.set_state(saved['collection_rng'])
            if 'mixing' in saved:
                self.mixing.load_state_dict(saved['mixing'])
            else:
                self.resume_migrations.append('reset coin allocation replaced with fixed randomized teacher population')
            self._restore_sampling_rng(saved)
            torch.set_rng_state(saved['torch_rng'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
        self.mixing.actor_updates = self.learner.updates['actor']
        self._write_json(self.output / 'configuration.json', asdict(config))
        self._write_json(self.output / 'resume-migrations.json', self.resume_migrations)

    def _initialize_course_progression(self, saved):
        config = self.config.course_progression
        if config is None:
            return
        self.course_progression = CourseProgression(config, self.env.adapter.curriculum)
        from ..course.metrics import install_jump_outcome_metrics
        install_jump_outcome_metrics(self.env.adapter.env)
        if saved is not None and saved.get('course_progression') is not None:
            self.course_progression.load_state_dict(saved['course_progression'])
        elif saved is not None:
            config.validate_activation(saved, self.config.course_coordinate)
        elif config.start_collection_step != 0:
            raise ValueError('Fresh course progression must begin at collection step zero')
        self.course_progression.apply_difficulty()

    def _initialize_guard(self, saved):
        config = self.config
        self.guard = (NativeRegressionGuard(config.guard_drop_fraction, config.guard_patience,
                                            config.learning.actor_learning_rate, window=config.guard_window,
                                            mode=config.guard_mode, min_rate_fraction=config.guard_min_rate_fraction,
                                            recover_after=config.guard_recover_after)
                      if config.guard_drop_fraction else None)
        self.guard_cold_cycles_remaining = 0
        self.guard_cold_previous_bad_cycles = None
        if getattr(self, 'guard_settings_changed', None):
            self.resume_migrations.append({'guard_settings_changed': self.guard_settings_changed})
        if saved is not None and self.guard is None:
            # A rate lowered by an earlier guard must not outlive the guard.
            for group in self.learner.optimizers['actor'].param_groups:
                group['lr'] = config.learning.actor_learning_rate
        if saved is not None and self.guard is not None:
            if saved.get('guard') is not None and not self.guard.load_state_dict(saved['guard']):
                self.resume_migrations.append('guard judgment restarted under a new window/mode')
            # The resumed Adam carries the rate of the saved guard state; apply this guard's rate.
            for group in self.learner.optimizers['actor'].param_groups:
                group['lr'] = self.guard.learning_rate()
            # Cold births can trigger a premature restore, but report-only
            # monitoring must keep observing them for human review. In that
            # mode an absolute task warmup endpoint must not become a new
            # hundreds-of-cycles blind period after every handoff.
            if config.policy_start_cycle and self.guard.restores:
                self.guard_cold_cycles_remaining = config.policy_start_cycle
                self.guard_cold_previous_bad_cycles = self.guard.bad_cycles
                self.guard.bad_cycles = 0

    def _check_saved(self, saved):
        if (saved.get('format') != CHECKPOINT_FORMAT or saved['model_contract'] != MODEL_CONTRACT
                or saved['schema_fingerprint'] != self.env.schema.fingerprint):
            raise ValueError('Online checkpoint contract differs')
        configuration, requested = asdict(OnlineConfig.from_dict(saved['config'])), asdict(self.config)
        before, after = configuration.pop('world_datasets'), requested.pop('world_datasets')
        self.world_datasets_changed = None if before == after else {'before': before, 'after': after}
        before, after = configuration.pop('actor_features'), requested.pop('actor_features')
        self.actor_features_activated = before != after
        if self.actor_features_activated and (before is not None or after is None):
            raise ValueError('Resume may activate actor reference features, not silently replace them')
        before, after = configuration.pop('chassis_command_contract'), requested.pop('chassis_command_contract')
        self.chassis_commands_activated = before != after
        if self.chassis_commands_activated and (before, after) != (LEGACY_COMMANDS, HEADING_COMMANDS):
            raise ValueError('Resume only activates the versioned chassis command projection')
        old_progression, new_progression = configuration.pop('course_progression'), requested.pop('course_progression')
        self.jump_height_progression_changed = None
        if old_progression is not None and new_progression is not None:
            # A sampling-clock choice does not replace the coordinate mapping
            # or its gate evidence. All other progression changes stay checked.
            before = old_progression.pop('jump_height_progression')
            after = new_progression.pop('jump_height_progression')
            if before != after:
                self.jump_height_progression_changed = {'before': before, 'after': after}
        self.course_progression_activated = old_progression != new_progression
        if self.course_progression_activated:
            if old_progression is not None or self.config.course_progression is None:
                raise ValueError('An active course progression cannot be replaced or removed on resume')
            self.config.course_progression.validate_activation(saved, self.config.course_coordinate)
        # The cycle budget and the guard's supervision settings may change on
        # resume; the guard does not change the gradient recipe it supervises.
        changed = {}
        for name in ('cycles', *GUARD_SETTINGS):
            before, after = configuration.pop(name), requested.pop(name)
            if name != 'cycles' and before != after:
                changed[name] = [before, after]
        if configuration != requested or self.config.cycles <= saved['cycle']:
            raise ValueError('Resume may extend the cycle budget or change guard supervision, not the training recipe')
        self.guard_settings_changed = changed
        if saved.get('teacher_source') != self.teacher_source:
            raise ValueError('Resume teacher source differs')
        if json.dumps(saved.get('stair_scenario'), sort_keys=True) != json.dumps(self.stair_scenario, sort_keys=True):
            raise ValueError('Resume stair protocol/source settings differ from the native environment')
        stored_jump = saved.get('jump_scenario')
        if ((stored_jump is None) != (self.jump_scenario is None)
                or stored_jump is not None and JumpTaskConfig.from_dict(stored_jump)
                != JumpTaskConfig.from_dict(self.jump_scenario)):
            raise ValueError('Resume jump protocol/reference settings differ from the native environment')
        if saved.get('recovery_scenario') != self.recovery_scenario:
            raise ValueError('Resume recovery lifecycle settings differ from the native environment')

    @staticmethod
    def _write_json(path, value):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def _restore_sampling_rng(self, saved):
        """A storage-device migration must supply explicitly initialized RNGs."""
        for name, generator in (('sample', self.generator), ('validation', self.validation_generator)):
            recorded = saved.get(name + '_rng_device')
            if (recorded is None and self.config.replay_device == 'cpu'
                    or recorded is not None and torch.device(recorded).type != generator.device.type):
                raise ValueError('Replay sampling RNG device differs; use an explicit storage migration')
            generator.set_state(saved[name + '_rng'])

    def sample_starts(self, batch):
        replay = self.collector.replay
        candidates = policy_start_candidates(replay, self.config, self.train_ids, self.env.policy_dt_s)
        start, frames = replay.sample_starts(batch, self.learner.actor.history_length,
                                             generator=self.generator, candidates=candidates)
        return {name: value.to(self.device) for name, value in start.items()}, frames.to(self.device)

    def sample_windows(self, batch, horizon):
        from .fitting import WorldModelFit

        replay = self.collector.replay
        names = WorldModelFit.RECORD_FIELDS + tuple(
            name for name in WorldModelFit.OPTIONAL_RECORD_FIELDS if name in replay.data)
        staging = None
        if replay.device.type == 'cpu' and self.device.type == 'cuda':
            key = (batch, horizon, names)
            if getattr(self, '_model_window_shape', None) != key:
                self._model_window_staging = {
                    name: torch.empty(batch, horizon, *replay.data[name].shape[2:],
                                      dtype=replay.data[name].dtype, pin_memory=True) for name in names}
                self._model_window_shape = key
            staging = self._model_window_staging
        windows = replay.sample_windows(batch, horizon, generator=self.generator, env_ids=self.train_ids,
                                        field_names=names, out=staging)
        moved = {name: value.to(self.device, non_blocking=staging is not None) for name, value in windows.items()}
        if staging is not None:
            # Wait once after all transfers, before reusing pinned host storage.
            torch.cuda.current_stream(self.device).synchronize()
        auxiliary = getattr(self, 'auxiliary_world', None)
        return moved if auxiliary is None else auxiliary.mix(moved)

    @torch.no_grad()
    def initialize_added_critic_norm(self):
        """Calibrate appended terrain/source features, preserving old statistics."""
        start, _ = self.sample_starts(self.config.imagination_batch_size)
        self.learner.calibrate_appended_critic_features(start, self.critic_norm_from)
        self.critic_norm_from = None

    @torch.no_grad()
    def initialize_task_norm(self):
        """Calibrate only newly introduced columns; retain every old critic statistic."""
        if not self.config.record_tasks or self.learner.critic.task_dim != MOTION_TASK_DIM:
            raise ValueError('Task normalization migration requires the command-aware critic')
        start, _ = self.sample_starts(self.config.imagination_batch_size)
        features = MotionTask.decode(start['motion_task']).critic_features(decode_physical(start['physical']))
        norm = self.learner.critic.norm
        norm.mean[-MOTION_TASK_DIM:].copy_(features.mean(0))
        norm.var[-MOTION_TASK_DIM:].copy_(features.var(0, unbiased=False))
        self.task_norm_pending = False

    @torch.no_grad()
    def refresh_contact_floor(self):
        """Refit imagination's contact-load floors to the current model on held-out rows."""
        learner = self.learner
        result = calibrate_contact_floor(learner.fit, self.collector.replay, self.validation_ids,
            self.config.validation_batch_size, self.validation_generator, self.device,
            threshold_n=learner.imagination.termination.limits.forbidden_body_load_n)
        learner.imagination.contact_floor = (result['body_floor'], result['base_floor'])
        return result['report']

    def save(self, *, with_replay):
        actor = {'format': CHECKPOINT_FORMAT + '-actor', 'model_contract': MODEL_CONTRACT,
                 'schema': self.env.schema.to_dict(), 'schema_fingerprint': self.env.schema.fingerprint,
                 'config': asdict(self.config), 'cycle': self.cycle, 'updates': self.learner.updates.copy(),
                 'resume_migrations': self.resume_migrations,
                 'actor': self.learner.actor.state_dict()}
        atomic_save(actor, self.output / f'actor-{self.cycle:06d}.pt')
        if not with_replay:
            return
        # Two rotating generations keep the preceding valid pair intact until
        # both files are durable. The pointer changes last; no old data is deleted.
        pointer = self.output / 'resume.json'
        previous = json.loads(pointer.read_text())['slot'] if pointer.exists() else 'b'
        slot = 'b' if previous == 'a' else 'a'
        replay_file = f'replay-{slot}.pt'
        atomic_save({'cycle': self.cycle, 'schema_fingerprint': self.env.schema.fingerprint,
                     'replay': self.collector.replay.state_dict()}, self.output / replay_file)
        payload = {**{key: value for key, value in actor.items() if key not in ('actor', 'updates')},
            'format': CHECKPOINT_FORMAT, 'learner': self.learner.state_dict(), 'replay_file': replay_file,
            'seed_source': self.seed_source, 'teacher_source': self.teacher_source,
            'actor_initialization': self.actor_initialization,
            'collection_steps': self.collector.steps, 'environment': self.env.training_state_dict(),
            'collection_rng': self.collector.generator.get_state(), 'mixing_rng': self.mixing.generator.get_state(),
            'mixing': self.mixing.state_dict(), 'resume_migrations': self.resume_migrations,
            'guard': None if self.guard is None else self.guard.state_dict(),
            'task_norm_pending': self.task_norm_pending,
            'course_progression': (None if self.course_progression is None else self.course_progression.state_dict()),
            'critic_norm_from': self.critic_norm_from, 'stair_scenario': self.stair_scenario,
            'jump_scenario': self.jump_scenario, 'recovery_scenario': self.recovery_scenario,
            'sample_rng': self.generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'validation_rng': self.validation_generator.get_state(),
            'sample_rng_device': str(self.generator.device),
            'validation_rng_device': str(self.validation_generator.device),
            'auxiliary_world': (None if getattr(self, 'auxiliary_world', None) is None
                                else self.auxiliary_world.state_dict()),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        checkpoint = f'resume-{slot}.pt'
        atomic_save(payload, self.output / checkpoint)
        self._write_json(pointer, {'slot': slot, 'checkpoint': checkpoint, 'cycle': self.cycle,
                                  'replay_file': replay_file, 'physics_resume': 'explicit cold reset; new episode namespace'})

    def train(self, *, checkpoint_requested=None):
        cfg, learner = self.config, self.learner
        started = time.monotonic()
        if self.cycle == 0:
            for _ in range(cfg.estimator_warmup_updates):
                start, frames = self.sample_starts(cfg.imagination_batch_size)
                learner.fit_estimator(start, frames)
        for cycle in range(self.cycle + 1, cfg.cycles + 1):
            tick = time.monotonic()
            teacher_before, student_before = self.mixing.teacher_steps, self.mixing.student_steps
            self.mixing.actor_updates = learner.updates['actor']
            # From the first policy cycle on; its collection still uses the initial
            # actor and becomes the baseline best.
            guard_deferred = self.guard is not None and self.guard_cold_cycles_remaining > 0
            guarded = self.guard is not None and cycle > cfg.policy_start_cycle and not guard_deferred
            snapshot = (self.guard.snapshot(learner.actor, learner.optimizers['actor'],
                        estimator_optimizer=learner.optimizers['estimator'])
                        if guarded and self.guard.restores else None)
            progression = getattr(self, 'course_progression', None)
            before_counters = (None if progression is None else
                               progression.snapshot_counters(self.env.adapter.env))
            collection = self.collector.collect(cfg.collection_steps, actor_updates=learner.updates['actor'])
            course = (None if progression is None else
                      progression.observe(self.env.adapter.env, collection, before_counters))
            guard = (self.guard.observe(collection.get('student_reward_mean'), snapshot, learner.actor,
                        learner.optimizers['actor'], estimator_optimizer=learner.optimizers['estimator'])
                     if guarded else None)
            if guard_deferred:
                self.guard_cold_cycles_remaining -= 1
                guard = {'student_reward': collection.get('student_reward_mean'),
                    'best_student_reward': self.guard.best_reward, 'bad_cycles': self.guard.bad_cycles,
                    'restored': False, 'restorations': self.guard.restorations,
                    'actor_learning_rate': learner.optimizers['actor'].param_groups[0]['lr'],
                    'deferred_reason': 'cold_resume',
                    'remaining_cycles': self.guard_cold_cycles_remaining,
                    'cold_resume_previous_bad_cycles': self.guard_cold_previous_bad_cycles}
            if getattr(self, 'task_norm_pending', False):
                self.initialize_task_norm()
            collected_at = time.monotonic()
            self._write_json(self.output / 'progress.json', {'cycle': cycle, 'phase': 'world', 'completed': 0,
                                                            'requested': cfg.model_updates})
            world = []
            auxiliary = getattr(self, 'auxiliary_world', None)
            auxiliary_before = {} if auxiliary is None else auxiliary.sampled.copy()
            for index in range(cfg.model_updates):
                windows = self.sample_windows(cfg.model_batch_size, cfg.fit_horizon)
                world.append(learner.fit_model(windows))
                if (index + 1) % 12 == 0 or index + 1 == cfg.model_updates:
                    self._write_json(self.output / 'progress.json', {'cycle': cycle, 'phase': 'world',
                        'completed': index + 1, 'requested': cfg.model_updates, 'updates': learner.updates.copy()})
            floor_report = self.refresh_contact_floor() if cfg.calibrate_contact_floor else None
            fitted_at = time.monotonic()
            policy = []
            policy_learning = cycle > cfg.policy_start_cycle
            critic_only = not policy_learning and cfg.critic_warmup
            if (policy_learning or critic_only) and getattr(self, 'critic_norm_from', None) is not None:
                self.initialize_added_critic_norm()
            for index in range(cfg.actor_updates):
                start, frames = self.sample_starts(cfg.imagination_batch_size)
                policy.append(learner.update_policy(start, frames, microbatch_size=cfg.imagination_microbatch_size,
                                                    train_actor=policy_learning)
                              if policy_learning or critic_only else learner.fit_estimator(start, frames))
                if (index + 1) % 4 == 0 or index + 1 == cfg.actor_updates:
                    self._write_json(self.output / 'progress.json', {'cycle': cycle,
                        'phase': ('policy' if policy_learning else
                                  'world_warmup_critic' if critic_only else 'world_warmup_estimator'),
                        'completed': index + 1, 'requested': cfg.actor_updates, 'updates': learner.updates.copy()})
            self.cycle = cycle
            record = {'cycle': cycle, 'real_transitions': self.collector.steps * self.env.num_envs,
                'policy_learning': policy_learning,
                'course_coordinate': collection.get('course_coordinate', cfg.course_coordinate),
                'replay_transitions': len(self.collector.replay), 'collection': collection,
                'world': {key: sum(row[key] for row in world) / len(world) for key in world[0]},
                'policy': {key: sum(row[key] for row in policy) / len(policy) for key in policy[0]},
                'updates': learner.updates.copy(), 'skipped': learner.skipped.copy(),
                **({} if auxiliary is None else {'auxiliary_world_windows': {
                    name: count - auxiliary_before[name] for name, count in auxiliary.sampled.items()}}),
                **({} if guard is None else {'guard': guard}),
                **({} if course is None else {'course': course}),
                **({} if floor_report is None else {'contact_floor': {
                    name: {key: channel[key] for key in ('floor_n', 'false_positive_raw', 'false_positive_floored',
                                                         'true_positive_raw', 'true_positive_floored')}
                    for name, channel in floor_report['channels'].items()}}),
                'teacher_target_fraction': self.mixing.fraction,
                'teacher_transitions': self.mixing.teacher_steps - teacher_before,
                'student_transitions': self.mixing.student_steps - student_before,
                'teacher_transitions_since_cold_start': self.mixing.teacher_steps,
                'student_transitions_since_cold_start': self.mixing.student_steps,
                'seconds': {'collection': collected_at - tick, 'world': fitted_at - collected_at,
                            'policy': time.monotonic() - fitted_at, 'elapsed': time.monotonic() - started},
                'cuda_peak_allocated_bytes': (torch.cuda.max_memory_allocated(self.device)
                                              if torch.device(self.device).type == 'cuda' else 0)}
            # Invalid optimizer attempts are logged as null, never nonstandard JSON NaN.
            for group in ('world', 'policy'):
                record[group] = {key: value if math.isfinite(value) else None for key, value in record[group].items()}
            with (self.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
            final = cycle == cfg.cycles
            handoff = checkpoint_requested is not None and checkpoint_requested()
            full = cycle % cfg.replay_checkpoint_interval == 0 or final or handoff
            if full or cycle % cfg.checkpoint_interval == 0:
                try:
                    validation = validate(learner.fit, self.collector.replay, self.validation_ids,
                        cfg.validation_horizon, cfg.validation_batch_size, self.validation_generator, self.device,
                        contact_threshold_n=learner.imagination.termination.limits.forbidden_body_load_n,
                        new_episode_namespace=self.collector.episode_offset,
                        termination=learner.imagination.termination)
                    if floor_report is not None:
                        validation['contact_floor'] = floor_report
                    self._write_json(self.output / f'world-validation-{cycle:06d}.json', validation)
                    if auxiliary is not None:
                        self._write_json(self.output / f'auxiliary-world-validation-{cycle:06d}.json',
                                         auxiliary.validate(learner, cfg, self.device))
                finally:
                    # A diagnostic failure must not discard the completed cycle
                    # that is needed to reproduce it. Preserve state, then raise.
                    self.save(with_replay=full)
            if handoff:
                break
        return {'cycle': self.cycle, 'updates': learner.updates, 'skipped': learner.skipped,
                'real_transitions': self.collector.steps * self.env.num_envs,
                'scope': ('complete checkpoint written for process handoff; training budget remains'
                          if self.cycle < cfg.cycles else
                          'online training budget finished; native capability evaluation still required')}
