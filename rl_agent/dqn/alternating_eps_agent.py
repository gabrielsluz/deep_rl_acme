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

"""DQN agent implementation."""

from typing import Sequence

from acme import specs
from acme.agents import agent
from acme.agents import replay
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.jax import utils

from dqn import config as dqn_config
from dqn import learning_lib
from dqn import losses
from dqn.actor import EpsilonPolicy, EpsilonActorState

import jax
import jax.numpy as jnp
import optax
import rlax


# Function similar to alternating_epsilons_actor_core but apply_and_sample receives an unbatched observation
def unbatched_alternating_epsilons_actor_core(
    policy_network: EpsilonPolicy, epsilons: Sequence[float],
) -> actor_core_lib.ActorCore[EpsilonActorState, None]:
  """Returns actor components for alternating epsilon exploration.
  Args:
    policy_network: A feedforward action selecting function.
    epsilons: epsilons to alternate per-episode for epsilon-greedy exploration.
  Returns:
    A feedforward policy.
  """
  epsilons = jnp.array(epsilons)

  def apply_and_sample(params: networks_lib.Params,
                       observation: networks_lib.Observation,
                       state: EpsilonActorState):
    random_key, key = jax.random.split(state.rng)
    observation = utils.add_batch_dim(observation)
    action = utils.squeeze_batch_dim(policy_network(params, key, observation, state.epsilon))
    return (action.astype(jnp.int64),
            EpsilonActorState(rng=random_key, epsilon=state.epsilon))

  def policy_init(random_key: networks_lib.PRNGKey):
    random_key, key = jax.random.split(random_key)
    epsilon = jax.random.choice(key, epsilons)
    return EpsilonActorState(rng=random_key, epsilon=epsilon)

  return actor_core_lib.ActorCore(
      init=policy_init, select_action=apply_and_sample,
      get_extras=lambda _: None)


class AltDQN(agent.Agent):
    """DQN agent. With alternating epsilons
    
    This implements a single-process DQN agent. This is a simple Q-learning
    algorithm that inserts N-step transitions into a replay buffer, and
    periodically updates its policy by sampling these transitions using
    prioritization.
    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network: networks_lib.FeedForwardNetwork,
        config: dqn_config.DQNConfig,
    ):
        """Initialize the agent."""
        # Data is communicated via reverb replay.
        reverb_replay = replay.make_reverb_prioritized_nstep_replay(
            environment_spec=environment_spec,
            n_step=config.n_step,
            batch_size=config.batch_size,
            max_replay_size=config.max_replay_size,
            min_replay_size=config.min_replay_size,
            priority_exponent=config.priority_exponent,
            discount=config.discount,
        )
        self._server = reverb_replay.server

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(config.learning_rate),
        )
        key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))
        # The learner updates the parameters (and initializes them).
        loss_fn = losses.PrioritizedDoubleQLearning(
            discount=config.discount,
            importance_sampling_exponent=config.importance_sampling_exponent,
        )
        learner = learning_lib.SGDLearner(
            network=network,
            loss_fn=loss_fn,
            data_iterator=reverb_replay.data_iterator,
            optimizer=optimizer,
            target_update_period=config.target_update_period,
            random_key=key_learner,
            replay_client=reverb_replay.client,
        )

        # The actor selects actions according to the policy.
        def policy(params: networks_lib.Params, key: jnp.ndarray,
                observation: jnp.ndarray, epsilon: float) -> jnp.ndarray:
            action_values = network.apply(params, observation)
            return rlax.epsilon_greedy(epsilon).sample(key, action_values)
        self._policy = policy

        actor_core = unbatched_alternating_epsilons_actor_core(policy, config.alternating_eps)
        variable_client = variable_utils.VariableClient(learner, '')
        actor = actors.GenericActor(
            actor_core, key_actor, variable_client, reverb_replay.adder)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(config.batch_size, config.min_replay_size),
            observations_per_step=config.observations_per_step,
        )


class AltDQNEval(agent.Agent):
    """
    Only for evaluating a policy.
    Sets epsilon to fixed value
    And sets learner parameters to high values to avoid learning.
    Still observes
    """
    def __init__(
        self,
        dqn: AltDQN,
        epsilon: float
    ):
        learner = dqn._learner
        _, key_actor = jax.random.split(jax.random.PRNGKey(1))
        actor_core = unbatched_alternating_epsilons_actor_core(dqn._policy, [epsilon])
        variable_client = variable_utils.VariableClient(learner, '')
        actor = actors.GenericActor(
            actor_core, key_actor, variable_client, None)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=int(1e10),
            observations_per_step=int(1e9),
        )