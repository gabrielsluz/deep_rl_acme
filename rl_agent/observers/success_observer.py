from typing import Dict

from acme.utils.observers import base
import dm_env
import numpy as np


class SuccessObserver(base.EnvLoopObserver):
  """An observer that checks succes in GymWrapper.get_info."""

  def __init__(self):
    self._success = False

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self._success = False

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    info_d = env.get_info()
    if 'success' in info_d:
        self._success = info_d['success']

  def get_metrics(self) -> Dict[str, base.Number]:
    """Returns metrics collected for the current episode."""
    return {'success': self._success}
