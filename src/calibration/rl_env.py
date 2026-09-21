"""Gymnasium training environment. Reward evaluation never enters observations."""
from __future__ import annotations
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .control import FeedbackSession, STATE_NAMES, evaluation_loss


class AdaptationEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, tasks, device, log_path=None):
        self.tasks, self.device = tasks, device
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(STATE_NAMES),), np.float32)
        self.completed = 0
        self.episode_returns = []
        self.log_file = open(log_path, 'w', encoding='utf-8') if log_path else None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Round-robin pseudo-subjects; random feedback order within each episode.
        self.task = self.tasks[self.completed % len(self.tasks)]
        order = self.np_random.permutation(9)
        self.session = FeedbackSession(self.task.source, self.task.feedback, order, self.device)
        self.before = evaluation_loss(self.session.adapter.model, self.task.evaluation)
        self.total_reward = 0.0
        self.initial_loss = self.before
        return self.session.prepare(), {}

    def step(self, action):
        record = self.session.act(int(action))
        after = evaluation_loss(self.session.adapter.model, self.task.evaluation)
        reward = self.before - after
        record.update(episode=self.completed + 1, pseudo_subject=self.task.subject,
                      reward=reward, eval_ce_before=self.before, eval_ce_after=after)
        self.before = after
        self.total_reward += reward
        if self.log_file:
            self.log_file.write(json.dumps(record) + '\n')
        done = len(self.session.visible) == 9
        if done:
            self.completed += 1
            assert abs(self.total_reward - (self.initial_loss - after)) < 1e-7
            self.episode_returns.append(dict(episode=self.completed, subject=self.task.subject,
                                            reward=self.total_reward, initial_ce=self.initial_loss,
                                            final_ce=after, update_steps=self.session.total_steps))
            if self.log_file:
                self.log_file.flush()
        obs = np.zeros(len(STATE_NAMES), np.float32) if done else self.session.prepare()
        return obs, reward, done, False, {}

    def close(self):
        if self.log_file:
            self.log_file.close()
