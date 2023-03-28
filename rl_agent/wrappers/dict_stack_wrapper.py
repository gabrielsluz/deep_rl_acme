from acme import types
from acme.wrappers import base
import dm_env
import numpy as np
from collections import deque

from wrappers.frame_stack_wrapper import FrameStack

# Class that concatenates the current observation with the previous ones
class VecStackConcat:
    def __init__(self, stackDepth=10, stateShape=(2,), dtype=np.float32):
        assert len(stateShape) == 1, (
            'Observation spec must be 1D, got {}'.format(stateShape))
        self.stack = deque(maxlen=stackDepth)
        self.stack_depth = stackDepth
        self.state_shape = stateShape
        self.frame_stack_shape = (self.state_shape[0]*self.stack_depth,)
        self.dtype = dtype
    
    def clearStack(self):
        self.stack.clear()
    
    def state(self, state):
        self.stack.append(state)
        return self.assemblyState()
    
    def assemblyState(self):
        state_stack = np.zeros(shape=self.frame_stack_shape, dtype=self.dtype)
        # Iterate over the deque in reverse order so that the most recent state is at the start
        state_stack[:self.state_shape[0]*len(self.stack)] = np.concatenate(
            list(reversed(self.stack)), axis=0)
        return state_stack

# DictStackWrapper: Does the same as FrameStackWrapper, but for dicts
class DictStackWrapper(base.EnvironmentWrapper):
  """
  Stacks the current observation with the previous ones.
  If an observation is 1D, then concatenates the current observation with the previous ones.
  """

  def __init__(self, environment: dm_env.Environment, stackDepth=10):
    super().__init__(environment)
    observation_spec = environment.observation_spec()
    for key in observation_spec:
        assert len(observation_spec[key].shape) in [1,2,3], (
            'Observation spec must be 1D, 2D or 3D, got {}'.format(observation_spec[key].shape))
    
    self._stacker = {}
    self._observation_spec = {}
    for key in observation_spec:
        if len(observation_spec[key].shape) == 1:
            self._stacker[key] = VecStackConcat(stackDepth, observation_spec[key].shape, observation_spec[key].dtype)
        else:
            self._stacker[key] = FrameStack(stackDepth, observation_spec[key].shape, observation_spec[key].dtype)
        self._observation_spec[key] = dm_env.specs.BoundedArray(
            shape=self._stacker[key].frame_stack_shape,
            dtype=observation_spec[key].dtype,
            minimum=-np.inf,
            maximum=np.inf,
            name=observation_spec[key].name)

  def step(self, action) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    new_obs = {}
    for key in timestep.observation:
        new_obs[key] = self._stacker[key].state(timestep.observation[key])
    return timestep._replace(observation=new_obs)

  def reset(self) -> dm_env.TimeStep:
    for key in self._stacker:
        self._stacker[key].clearStack()
    timestep = self._environment.reset()
    new_obs = {}
    for key in timestep.observation:
        new_obs[key] = self._stacker[key].state(timestep.observation[key])
    return timestep._replace(observation=new_obs)

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec