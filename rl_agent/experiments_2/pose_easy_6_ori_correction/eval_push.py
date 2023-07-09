"""
experiments/pose_easy_6 but with new config and bug correction in orientation
    'aux_info': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)})
    def getObservation(self):	
        obs = {}	
        obs['state_img'] = self.push_simulator.getStateImg()	
        obs['aux_info'] = np.zeros(shape=(2,), dtype=np.float32)	
        obs['aux_info'][0] = self.push_simulator.distToObjective() / self.max_objective_dist	
        obs['aux_info'][1] = self.push_simulator.distToOrientation() / np.pi	
        return obs
"""

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
from wrappers.dict_stack_wrapper import DictStackWrapper
from push_environment_loop import PushEnvironmentLoop
from research_envs.envs.box2D_img_pushing_pose_env import Box2DPushingEnv, Box2DPushingEnvConfig
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulatorConfig
from research_envs.envs.rewards import RewardFunctions
from observers.success_observer import SuccessObserver
from acme.utils.loggers import CSVLogger

# Utils
# def calc_suc_rate(data: list) -> float:
#     suc_cnt = 0
#     for i in data:
#         suc_cnt += i['success']
#     return suc_cnt / len(data)

env_config = Box2DPushingEnvConfig(
    terminate_obj_dist = 12.0,
    goal_dist_tol = 2.0,
    goal_ori_tol= np.pi / 36,
    max_steps = 200,
    # Reward config:
    reward_fn_id = RewardFunctions.PROGRESS,
    # Push simulator config:
    push_simulator_config = PushSimulatorConfig(
        pixels_per_meter=20, width=1024, height=1024,
        obj_proximity_radius=12.0,
        objTuple=(
            {'name':'Circle', 'radius':4.0},
            {'name': 'Rectangle', 'height': 10.0, 'width': 5.0},
            {'name': 'Polygon', 'vertices': [(5,10), (0,0), (10,0)]},
        ),
        max_dist_obj_goal = 30,
        min_dist_obj_goal = 2,
        max_ori_obj_goal = np.pi / 2
    )
)

# ENV
def create_environment():
    env = Box2DPushingEnv(config=env_config)
    env = GymWrapper(env)
    env = DictStackWrapper(env, stackDepth=4)
    return env

def run(exp_num):
    jax.config.update('jax_enable_x64', True)

    env = create_environment()
    env_spec = specs.make_environment_spec(env)

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
            jax.nn.relu,
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

    agent = dqn.DQN(
        environment_spec=env_spec, 
        network=network, 
        batch_size=2048,
        # prefetch_size=4,
        target_update_period=2000,
        observations_per_step=40.0,
        min_replay_size=2048,
        max_replay_size=130000,
        # importance_sampling_exponent=0.2,
        # priority_exponent=0.6,
        # n_step=5,
        epsilon_start=1.0,
        epsilon_end=0.005,
        epsilon_decay_episodes=20*100,
        learning_rate=1e-3,
        discount=0.99,
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
        epsilon=0.005
    )
    eval_eps = 200

    eval_logs_file = open("eval_push_logs_{}.txt".format(exp_num), "a")
    logger = CSVLogger(directory_or_file=eval_logs_file)
    print("Eval Epoch")
    loop = PushEnvironmentLoop(env, agent_eval, logger=logger, observers=observers)
    loop.run(num_episodes=eval_eps)
    # suc_rate = calc_suc_rate(loop._logger.data[-eval_eps:])
    # print('EVAL in {} episodes: Success Rate: {:.3f}'.format(eval_eps, suc_rate))
    eval_logs_file.close()

if __name__ == '__main__':
    # Pass experiment number as argument
    run(sys.argv[1])
    print("End of code")
