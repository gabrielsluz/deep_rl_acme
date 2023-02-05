import haiku as hk
import jax.numpy as jnp

from acme.jax import networks as networks_lib
from acme import specs

import gym
from gym26_wrapper import GymWrapper

import dqn

import random
import numpy as np

def create_environment():
    env = gym.make("CartPole-v1")
    env = GymWrapper(env)
    return env

def main():
    env = create_environment()
    env_spec = specs.make_environment_spec(env)
    dummy_obs = jnp.zeros((4,))
    print(env_spec)

    # Rodar o Env por 
    n_episodes = 3
    for ep in range(n_episodes):
        print('Episode '+str(ep))
        timestep = env.reset()
        while not timestep.last():
            action = env.action_space.sample()
            timestep = env.step(action)
            print(timestep)

if __name__ == '__main__':
    main()