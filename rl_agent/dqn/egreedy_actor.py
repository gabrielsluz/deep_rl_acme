from typing import Generic, Optional, Sequence

from acme import adders
from acme import core
from acme import types
from acme.agents.jax import actor_core
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import jax
import jax.numpy as jnp

from dqn.actor import Epsilon, EpsilonActorState, EpsilonPolicy


def batched_epsilon_actor_core(
    policy_network: EpsilonPolicy,
) -> actor_core.ActorCore[EpsilonActorState, None]:
  """Returns actor core to use with EGreedyActor
  Args:
    policy_network: A feedforward action selecting function. Receives batch of observations.
  Returns:
    actor_core.ActorCore
  """
  def apply_and_sample(params: network_lib.Params,
                       observation: network_lib.Observation,
                       state: EpsilonActorState):
    random_key, key = jax.random.split(state.rng)
    observation = utils.add_batch_dim(observation)
    action = utils.squeeze_batch_dim(policy_network(params, key, observation, state.epsilon))
    return (action.astype(jnp.int64),
            EpsilonActorState(rng=random_key, epsilon=state.epsilon))

  def policy_init(random_key: network_lib.PRNGKey):
    random_key, _ = jax.random.split(random_key)
    return EpsilonActorState(rng=random_key, epsilon=0.0)

  return actor_core.ActorCore(
      init=policy_init, select_action=apply_and_sample,
      get_extras=lambda _: None)


class EGreedyActor(core.Actor, Generic[actor_core.State, actor_core.Extras]):
  """An actor implemented based on GenericActor, that uses E-Greedy exploration 
  with linear decayment of epsilon. The decayment happens at the start of every episode.
  Epsilon linearly decays from epsilon_start to epsilon_end in epsilon_decay_episodes episodes.

  An actor based on a policy which takes observations and EpsilonActorState and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """
  def __init__(
      self,
      actor: actor_core.ActorCore[EpsilonActorState, actor_core.Extras],
      epsilon_start: Epsilon,
      epsilon_end: Epsilon,
      epsilon_decay_episodes: int,
      random_key: network_lib.PRNGKey,
      variable_client: Optional[variable_utils.VariableClient],
      adder: Optional[adders.Adder] = None,
      jit: bool = True,
      backend: Optional[str] = 'cpu',
      per_episode_update: bool = False
  ):
    """Initializes a feed forward actor.
    Args:
      actor: actor core.
      epsilon_start: float in [0,1] that gives the starting value of epsilon.
      epsilon_decay_episodes: Epsilon linearly decays from epsilon_start to epsilon_end
        in epsilon_decay_episodes episodes.
      random_key: Random key.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      jit: Whether or not to jit the passed ActorCore's pure functions.
      backend: Which backend to use when jitting the policy.
      per_episode_update: if True, updates variable client params once at the
        beginning of each episode
    """
    self._random_key = random_key
    self._variable_client = variable_client
    self._adder = adder
    self._state = None

    # Unpack ActorCore, jitting if requested.
    if jit:
      self._init = jax.jit(actor.init, backend=backend)
      self._policy = jax.jit(actor.select_action, backend=backend)
    else:
      self._init = actor.init
      self._policy = actor.select_action
    self._get_extras = actor.get_extras
    self._per_episode_update = per_episode_update

    self.epsilon = epsilon_start
    self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes
    self.epsilon_end = epsilon_end

  @property
  def _params(self):
    return self._variable_client.params if self._variable_client else []

  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:
    action, self._state = self._policy(self._params, observation, self._state)
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._random_key, key = jax.random.split(self._random_key)
    self._state = self._init(key)

    # Epsilon Decay
    if self.epsilon_end < (self.epsilon - self.epsilon_decay):
      self._state.epsilon = self.epsilon
      self.epsilon = self.epsilon - self.epsilon_decay

    if self._adder:
      self._adder.add_first(timestep)
    if self._variable_client and self._per_episode_update:
      self._variable_client.update_and_wait()

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(
          action, next_timestep, extras=self._get_extras(self._state))

  def update(self, wait: bool = False):
    if self._variable_client and not self._per_episode_update:
      self._variable_client.update(wait)