from acme import types
from acme.wrappers import base
import dm_env
import numpy as np
from collections import deque

class FrameStack:
    def __init__(self, frameStackDepth=10, stateShape=(3,3), dtype=np.float32):
        self.frame_stack = deque(maxlen=frameStackDepth)
        self.state_shape = stateShape
        self.stack_depth = frameStackDepth
        self.frame_stack_shape = (self.state_shape[0],self.state_shape[1], self.stack_depth)
        self.dtype = dtype

    def clearStack(self):
        self.frame_stack.clear()

    def state(self, state):
        self.frame_stack.append(state)
        return self.assemblyState()

    def assemblyState(self):
        state_stack = np.zeros(shape=self.frame_stack_shape, dtype=self.dtype)
        for i, state in enumerate(self.frame_stack):
            state_stack[:,:,i] = state
        return state_stack


class FrameStackWrapper(base.EnvironmentWrapper):
  """Stacks the current observation with the previous ones
  Shape (16,16) -> (16,16,frameStackDepth)
  """

  def __init__(self, environment: dm_env.Environment, frameStackDepth=10):
    super().__init__(environment)
    observation_spec = environment.observation_spec()
    assert len(observation_spec.shape) == 2
    
    self._stacker = FrameStack(frameStackDepth, observation_spec.shape, observation_spec.dtype)

    self._observation_spec = dm_env.specs.BoundedArray(
        shape=self._stacker.frame_stack_shape,
        dtype=observation_spec.dtype,
        minimum=-np.inf,
        maximum=np.inf,
        name='observation')

  def step(self, action) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._stacker.state(timestep.observation))

  def reset(self) -> dm_env.TimeStep:
    self._stacker.clearStack()
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._stacker.state(timestep.observation))

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec
