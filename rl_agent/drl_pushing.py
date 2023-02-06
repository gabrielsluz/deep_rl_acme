import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import acme
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

import dqn

from acme.wrappers import GymWrapper
from wrappers.add_channel_wrapper import AddChannelDimWrapper
from environment_loop import EnvironmentLoop
from research_envs.envs.box2D_img_pushing_env import Box2DPushingEnv
from research_envs.envs.rewards import RewardFunctions
from observers.success_observer import SuccessObserver
from acme.utils.loggers import InMemoryLogger

# Utils
def calc_suc_rate(data: list) -> float:
    suc_cnt = 0
    for i in data:
        suc_cnt += i['success']
    return suc_cnt / len(data)

# ENV
def create_environment():
    env = Box2DPushingEnv(smoothDraw=False, reward=RewardFunctions.PROJECTION, max_steps=100)
    env = GymWrapper(env)
    env = AddChannelDimWrapper(env)
    return env

def main():
    jax.config.update('jax_enable_x64', True)

    env = create_environment()
    env_spec = specs.make_environment_spec(env)
    # print(env_spec)
    
    # Calculate how big the last layer should be based on total # of actions.
    action_spec = env_spec.actions
    action_size = np.prod(action_spec.shape, dtype=int)
    # print(action_spec, action_size)

    # AGENT
    def network_fn(obs):
        network = hk.Sequential([
            hk.Conv2D(output_channels=8, kernel_shape=[4, 4], stride=4, padding='valid'),
            jax.nn.relu,
            hk.Conv2D(output_channels=16, kernel_shape=[3, 3], padding='valid'),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(env_spec.actions.num_values)
        ])
        x = obs
        x = network(x)
        return x

    dummy_action = utils.zeros_like(env_spec.actions)
    dummy_obs = utils.add_batch_dim(utils.zeros_like(env_spec.observations))

    mlp = hk.without_apply_rng(hk.transform(network_fn))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: mlp.init(rng, dummy_obs),
        apply=mlp.apply
    )

    agent = dqn.DQN(
        environment_spec=env_spec, 
        network=network, 
        batch_size=2048,
        # prefetch_size=4,
        # target_update_period=100,
        observations_per_step=50.0,
        min_replay_size=2048,
        # max_replay_size=1000000,
        # importance_sampling_exponent=0.2,
        # priority_exponent=0.6,
        # n_step=1,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=2000,
        learning_rate=1e-3,
        discount=0.95,
        # seed=1,
    )
    observers = [
        SuccessObserver()
    ]
    logger = InMemoryLogger()

    loop = EnvironmentLoop(env, agent, logger=logger, observers=observers)
    ep_per_epoch = 20
    for epoch_i in range(200):
        loop.run(num_episodes=ep_per_epoch)
        suc_rate = calc_suc_rate(loop._logger.data[-ep_per_epoch:])
        print('Epoch {}: Success Rate: {:.3f}'.format(epoch_i, suc_rate))

    """
    Faltando:
        Concatenar frames => FrameStack => Wrapper do Env. Talvez fazer um wrapper no Gym e usar o
            GymWrapper.
        Checar:
            - Rede Neural funciona, recebe entradas na dimensao correta, passa o batch certinho?
            - Tamanho do batch
            - Steps de learner por steps de actor => 1 learner por episodio => 1 por 20-100 observations?
            - Atualizar a target a cada 5 epocas
            

    """

if __name__ == '__main__':
    main()
    print("End of code")