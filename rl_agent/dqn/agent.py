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

from dqn import config as dqn_config
from dqn import learning_lib
from dqn import losses
from dqn.egreedy_actor import EGreedyActor, batched_epsilon_actor_core

import jax
import jax.numpy as jnp
import optax
import rlax
import numpy as np


class DQNFromConfig(agent.Agent):
    """DQN agent.
    
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

        actor_core = batched_epsilon_actor_core(policy)
        variable_client = variable_utils.VariableClient(learner, '')
        actor = EGreedyActor(
            actor_core, 
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay_episodes,
            key_actor, variable_client, reverb_replay.adder)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(config.batch_size, config.min_replay_size),
            observations_per_step=config.observations_per_step,
        )
    # Agent -> EGreedyActor -> ActorCore
    # Interagir com o agent => set_epsilon/get_epsilon
    # Pode usar o Agent normal, pois o actor recebe a observation e o ActorCore que recebe o state.
    # EGreedyActor => Ajustar funcoes para manter o epsilon e dacair em uma taxa
    # ActorCore => política deve receber estado como entrada => EpsilonState. 
    #   Com base na  alternating_epsilons_actor_core


class DQN(DQNFromConfig):
  """DQN agent.

  We are in the process of migrating towards a more modular agent configuration.
  This is maintained now for compatibility.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: networks_lib.FeedForwardNetwork,
      batch_size: int = 256,
      max_gradient_norm: float = np.inf,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      observations_per_step: float = 1.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon_start: float = 1.0,
      epsilon_end: float = 0.05,
      epsilon_decay_episodes: int = 100,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      seed: int = 1,
  ):
    config = dqn_config.DQNConfig(
        batch_size=batch_size,
        max_gradient_norm=max_gradient_norm,
        prefetch_size=prefetch_size,
        target_update_period=target_update_period,
        observations_per_step=observations_per_step,
        min_replay_size=min_replay_size,
        max_replay_size=max_replay_size,
        importance_sampling_exponent=importance_sampling_exponent,
        priority_exponent=priority_exponent,
        n_step=n_step,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        learning_rate=learning_rate,
        discount=discount,
        seed=seed,
    )
    super().__init__(
        environment_spec=environment_spec,
        network=network,
        config=config,
    )



class DQNEval(agent.Agent):
    """
    Only for evaluating a policy.
    Sets epsilon to fixed value
    And sets learner parameters to high values to avoid learning.
    Still observes
    """
    def __init__(
        self,
        dqn: DQN,
        epsilon: float
    ):
        actor = dqn._actor # EGreedyActor
        actor.epsilon = epsilon
        actor.epsilon_decay = 0.0
        actor.epsilon_end = epsilon
        learner = dqn._learner
        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=int(1e10),
            observations_per_step=int(1e9),
        )