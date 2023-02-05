import numpy as np


import jax
import jax.numpy as jnp
import haiku as hk

import gym
import acme
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

import dqn

from gym26_wrapper import GymWrapper
from environment_loop import EnvironmentLoop


# ENV
def create_environment():
    env = gym.make("CartPole-v1")
    env = GymWrapper(env)
    return env

def main():
    jax.config.update('jax_enable_x64', True)

    env = create_environment()
    env_spec = specs.make_environment_spec(env)
    print(env_spec)
    # Calculate how big the last layer should be based on total # of actions.
    action_spec = env_spec.actions
    action_size = np.prod(action_spec.shape, dtype=int)
    print(action_spec, action_size)
    # print(env.reset())

    # AGENT
    def network_fn(obs):
        x = obs
        x = hk.nets.MLP(output_sizes=[64, 64, env_spec.actions.num_values])(x)
        return x

    dummy_action = utils.zeros_like(env_spec.actions)
    dummy_obs = utils.zeros_like(env_spec.observations)

    mlp = hk.without_apply_rng(hk.transform(network_fn))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: mlp.init(rng, dummy_obs),
        apply=mlp.apply
    )

    agent = dqn.DQN(
        environment_spec=env_spec, 
        network=network, 
        # batch_size=64,
        # prefetch_size=4,
        # target_update_period=100,
        # samples_per_insert=1.0,
        # min_replay_size=1000,
        # max_replay_size=1000000,
        # importance_sampling_exponent=0.2,
        # priority_exponent=0.6,
        # n_step=1,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=100,
        # learning_rate=1e-3,
        # discount=0.95,
        # seed=1,
    )

    # Usar o estilo Alysson: Rodar 20 episodios por epoca e pegar os resultados

    loop = EnvironmentLoop(env, agent)
    loop.run(num_episodes=2)

if __name__ == '__main__':
    main()
    print("End of code")