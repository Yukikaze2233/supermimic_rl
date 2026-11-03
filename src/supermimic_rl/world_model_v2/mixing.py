"""Teacher/student environment allocation with changes only at episode boundaries.

An independent coin on every reset overrepresents the longer-lived teacher:
short student episodes repeatedly switch to a teacher that then occupies the
row for much longer. A fixed randomized row priority preserves the requested
population while avoiding a controller switch inside a physical episode.
"""
import torch


class MixedDecision:
    def __init__(self, environment, actor, teacher_decision, *, seed, initial_fraction, hold_updates, fade_updates):
        if not 0 <= initial_fraction <= 1 or hold_updates < 0 or fade_updates < 1:
            raise ValueError('Invalid teacher data schedule')
        self.env, self.actor, self.teacher = environment, actor, teacher_decision
        self.initial_fraction, self.hold_updates, self.fade_updates = initial_fraction, hold_updates, fade_updates
        self.actor_updates = 0
        self.generator = torch.Generator(device=environment.device).manual_seed(seed)
        self.priority = torch.randperm(environment.num_envs, device=environment.device, generator=self.generator)
        self.episode = torch.full_like(environment.current.episode_id, -1)
        self.teacher_rows = torch.zeros_like(self.episode, dtype=torch.bool)
        self.teacher_steps = self.student_steps = 0

    @property
    def fraction(self):
        return self.initial_fraction * max(0., 1 - max(0, self.actor_updates - self.hold_updates) / self.fade_updates)

    def state_dict(self):
        # Physical episodes are explicitly restarted on training resume. Only
        # the stable allocation and RNG survive; episode IDs are never replayed.
        return {'format': 'teacher-population-v1', 'priority': self.priority, 'rng': self.generator.get_state()}

    def load_state_dict(self, state):
        priority = state['priority'].to(device=self.env.device)
        expected = torch.arange(self.env.num_envs, device=self.env.device)
        if (state.get('format') != 'teacher-population-v1' or priority.shape != expected.shape
                or not torch.equal(priority.sort().values, expected)):
            raise ValueError('Teacher allocation differs from the native environment rows')
        self.priority = priority.long()
        self.generator.set_state(state['rng'])

    @torch.no_grad()
    def __call__(self, history):
        episode = self.env.current.episode_id
        fresh = episode != self.episode
        draw = self.priority < round(self.env.num_envs * self.fraction)
        self.teacher_rows = torch.where(fresh, draw, self.teacher_rows)
        self.episode = episode.clone()
        action, estimate = self.actor(history), self.actor.estimate(history)
        if self.teacher_rows.any():
            teacher_action, teacher_estimate = self.teacher()
            action = torch.where(self.teacher_rows[:, None], teacher_action, action)
            estimate = torch.where(self.teacher_rows[:, None], teacher_estimate, estimate)
        self.teacher_steps += int(self.teacher_rows.sum())
        self.student_steps += int((~self.teacher_rows).sum())
        return action, estimate
