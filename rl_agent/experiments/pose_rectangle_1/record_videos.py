# Record videos from the Eval episodes
# Include path two levels up
import sys
sys.path.append('../..')

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle

import acme
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

import dqn

from acme.wrappers import GymWrapper
# from wrappers.add_channel_wrapper import AddChannelDimWrapper
from wrappers.frame_stack_wrapper import FrameStackWrapper
from wrappers.record_video_wrapper import RecordVideoWrapper
from environment_loop import EnvironmentLoop
from research_envs.envs.box2D_img_pushing_pose_env import Box2DPushingEnv
from research_envs.envs.rewards import RewardFunctions
from observers.success_observer import SuccessObserver
from acme.utils.loggers import CSVLogger

# Utils
# def calc_suc_rate(data: list) -> float:
#     suc_cnt = 0
#     for i in data:
#         suc_cnt += i['success']
#     return suc_cnt / len(data)

# ENV
def create_environment():
    env = Box2DPushingEnv(smoothDraw=False, reward=RewardFunctions.PROJECTION, max_steps=200)
    env = RecordVideoWrapper(env, 'videos')
    env = GymWrapper(env)
    env = FrameStackWrapper(env, frameStackDepth=4)
    return env

def run(exp_num):
    jax.config.update('jax_enable_x64', True)

    env = create_environment()
    env_spec = specs.make_environment_spec(env)

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
        observations_per_step=40.0,
        min_replay_size=2048,
        max_replay_size=60000,
        # importance_sampling_exponent=0.2,
        # priority_exponent=0.6,
        # n_step=4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_episodes=20*50,
        learning_rate=1e-3,
        discount=0.95,
        # seed=1,
    )
    observers = [
        SuccessObserver()
    ]
    
    # Load model model
    # agent._learner.restore(savers.restore_from_path('learner_checkpoint'))
    with open(f'learner_checkpoint_{exp_num}', 'rb') as f:
        agent._learner.restore(pickle.load(f))

    # Eval
    agent_eval = dqn.DQNEval(
        dqn=agent,
        epsilon=0.05
    )
    eval_eps = 100

    eval_logs_file = open("eval_load_video_logs_{}.txt".format(exp_num), "a")
    logger = CSVLogger(directory_or_file=eval_logs_file)
    print("Eval Epoch")
    loop = EnvironmentLoop(env, agent_eval, logger=logger, observers=observers)
    loop.run(num_episodes=eval_eps)
    # suc_rate = calc_suc_rate(loop._logger.data[-eval_eps:])
    # print('EVAL in {} episodes: Success Rate: {:.3f}'.format(eval_eps, suc_rate))
    eval_logs_file.close()

if __name__ == '__main__':
    # Pass experiment number as argument
    run(sys.argv[1])
    print("End of code")