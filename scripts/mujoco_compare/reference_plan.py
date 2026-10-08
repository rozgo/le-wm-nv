"""Reference-dependent control work cached without changing live feedback."""
import numpy as np
from .environment import DT, reference, geometric


class ReferencePlan:
    def __init__(self, steps, trim, kind, heading, horizon=15):
        self.trim, self.horizon = trim, horizon
        trajectory = [reference(i*DT, kind, heading=heading) for i in range(steps+horizon+1)]
        self.states = np.array([s for s, _ in trajectory])
        self.accelerations = np.array([a for _, a in trajectory])
        self.feedforward = np.array([geometric(s, s, a, trim) for s, a in trajectory], np.float32)

    def at(self, state, step):
        actions = self.feedforward[step:step+self.horizon].copy()
        actions[0] = geometric(state, self.states[step], self.accelerations[step], self.trim)
        return actions, self.states[step+1:step+1+self.horizon].astype(np.float32)
