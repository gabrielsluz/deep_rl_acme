# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper that implements concatenation of observation fields."""

from typing import Sequence, Optional

from acme import types
from acme.wrappers import base
import dm_env
import numpy as np
import tree



class AddChannelDimWrapper(base.EnvironmentWrapper):
  """Adds a chanel dimension: 
  Shape (16,16) -> (16,16,1)
  """

  def __init__(self, environment: dm_env.Environment):
    """Initializes a new ConcatObservationWrapper.

    Args:
      environment: Environment to wrap.
    """
    super().__init__(environment)
    observation_spec = environment.observation_spec()
    assert len(observation_spec.shape) == 2

    dummy_obs = np.zeros(observation_spec.shape)
    dummy_obs = self._convert_observation(dummy_obs)

    self._observation_spec = dm_env.specs.BoundedArray(
        shape=dummy_obs.shape,
        dtype=observation_spec.dtype,
        minimum=-np.inf,
        maximum=np.inf,
        name='observation')

  def _convert_observation(self, observation):
    return observation[..., None]

  def step(self, action) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec
