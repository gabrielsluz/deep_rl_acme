import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import cv2

import acme
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

import dqn

from acme.wrappers import GymWrapper
# from wrappers.add_channel_wrapper import AddChannelDimWrapper
from wrappers.frame_stack_wrapper import FrameStackWrapper
from wrappers.dict_stack_wrapper import DictStackWrapper
from environment_loop import EnvironmentLoop
from research_envs.envs.box2D_img_dist_pushing_env import Box2DPushingEnv
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
    env = Box2DPushingEnv(smoothDraw=False, reward=RewardFunctions.PROGRESS, max_steps=200)
    env = GymWrapper(env)
    env = DictStackWrapper(env, stackDepth=4)
    return env

def main():
    jax.config.update('jax_enable_x64', True)

    env = create_environment()
    env_spec = specs.make_environment_spec(env)
    # print(env_spec)
    # Run a few steps and check consistency in dictstack
    # timestep = env.reset()
    # last_obs = timestep.observation['state_img']
    # last_aux = timestep.observation['aux_info']
    # for i in range(10):
    #     timestep = env.step(0)
    #     obs = timestep.observation['state_img']
    #     error = 0.0
    #     for j in range(obs.shape[2]-1):
    #         error += (obs[:,:,j+1] - last_obs[:,:,j]).sum()
    #     print(i, 'error state_img:', error)
    #     aux = timestep.observation['aux_info']
    #     error = 0.0
    #     for j in [0, 2, 4]:
    #         error += (aux[j+2:j+4] - last_aux[j:j+2]).sum()
    #     print(i, 'error aux_info:', error)
    #     last_aux = aux
    #     last_obs = obs
        # tst_img = (obs[:,:,0] * 255).astype(np.uint8)
        # cv2.imwrite(f'img{i+1}.png', tst_img)
    # for i in range(obs.shape[2]):
    #     tst_img = (obs[:,:,i] * 255).astype(np.uint8)
    #     cv2.imwrite(f'img{i+11}.png', tst_img)
    
    # Calculate how big the last layer should be based on total # of actions.
    action_spec = env_spec.actions
    action_size = np.prod(action_spec.shape, dtype=int)
    # print(action_spec, action_size)


    # AGENT
    def network_fn(obs):
        img_net = hk.Sequential([
            hk.Conv2D(output_channels=6, kernel_shape=[4, 4], stride=1, padding='VALID'),
            jax.nn.relu,
            hk.MaxPool(window_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),
            hk.Conv2D(output_channels=16, kernel_shape=[4, 4], stride=1, padding='VALID'),
            jax.nn.relu,
            hk.MaxPool(window_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),
            hk.Flatten()
        ])
        aux_net = hk.Sequential([
            hk.Linear(8),
            jax.nn.relu
        ])
        final_net = hk.Sequential([
            hk.Linear(120),
            jax.nn.relu,
            hk.Linear(84),
            jax.nn.relu,
            hk.Linear(8)
        ])
        img_emb = img_net(obs['state_img'])
        aux_emb = aux_net(obs['aux_info'])
        x = jnp.concatenate([img_emb, aux_emb], axis=1)
        x = final_net(x)
        return x

    dummy_action = utils.zeros_like(env_spec.actions)
    dummy_obs = utils.add_batch_dim(utils.zeros_like(env_spec.observations))

    mlp = hk.without_apply_rng(hk.transform(network_fn))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: mlp.init(rng, dummy_obs),
        apply=mlp.apply
    )

    params = mlp.init(0, dummy_obs)

    total_params = 0

    for key in params.keys():
        for inner_key in params[key].keys():
            print(key, inner_key, params[key][inner_key].shape)
            total_params += np.prod(params[key][inner_key].shape)
    print('Total number of params = ', total_params)

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
        # n_step=4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=20*130,
        learning_rate=1e-3,
        discount=0.95,
        # seed=1,
    )
    observers = [
        SuccessObserver()
    ]
    logger = InMemoryLogger()

    loop = EnvironmentLoop(env, agent, logger=logger, observers=observers)
    ep_per_epoch = 2
    for epoch_i in range(1):
        loop.run(num_episodes=ep_per_epoch)
        suc_rate = calc_suc_rate(loop._logger.data[-ep_per_epoch:])
        print('Epoch {}: Success Rate: {:.3f}'.format(epoch_i, suc_rate))

if __name__ == '__main__':
    main()
    print("End of code")