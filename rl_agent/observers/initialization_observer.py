from typing import Dict

from acme.utils.observers import base
import dm_env
import numpy as np


class InitializationObserver(base.EnvLoopObserver):
  """An observer that checks succes in GymWrapper.get_info."""

  def __init__(self):
    self._init_d = {}

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self._init_d['object'] = env.push_simulator.obj.obj_type
    self._init_d['init_pos_dist'] = env.push_simulator.distToObjective() * len(env.goal_l) # Dirty Hack
    self._init_d['init_ori_dist'] = env.push_simulator.distToOrientation() * len(env.goal_l) # Dirty Hack
    self._init_d['n_subgoals'] = len(env.goal_l)

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    pass

  def get_metrics(self) -> Dict[str, base.Number]:
    """Returns metrics collected for the current episode."""
    return self._init_d
