# Include path two levels up
import sys
sys.path.append('../..')

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import rlax
import pickle
import time
import collections

import acme
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

import dqn
from dqn.egreedy_actor import EGreedyActor, batched_epsilon_actor_core

from acme.wrappers import GymWrapper
from wrappers.dict_stack_wrapper import DictStackWrapper
from wrappers.record_video_wrapper import RecordVideoWrapper
from push_environment_loop import PushEnvironmentLoop
from research_envs.experiment_envs.pose_subgoal_env import PoseSubGoalEnv, PoseSubGoalEnvConfig
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulatorConfig
from observers.success_observer import SuccessObserver
from observers.initialization_observer import InitializationObserver
from acme.utils.loggers import CSVLogger

class ParamHolder():
    def __init__(self, params):
        self.params = params

# Eval loop
# episode_length,episode_return,episodes,steps,steps_per_second,success
def run_episode(env, agent, observers=[], max_steps=200, stuck_steps=10):
    res = {
        'episode_length': 0,
        'episode_return': 0,
        'stuck_count': 0,
    }
    obj_pos_deque = collections.deque(maxlen=stuck_steps)

    start_time = time.time()
    timestep = env.reset()
    agent.observe_first(timestep)
    for observer in observers:
      observer.observe_first(env, timestep)
    obj_pos_deque.append(np.array(env.push_simulator.getObjPosition()))

    while not timestep.last():
        # If the object is stuck, take action torwards the object
        if len(obj_pos_deque) == stuck_steps and np.allclose(obj_pos_deque[0], obj_pos_deque[-1]):
            action = env.push_simulator.getClosestActionToObject()
            res['stuck_count'] += 1
        else:
            # Generate an action from the agent's policy and step the environment.
            action = agent.select_action(timestep.observation)
        timestep = env.step(action)

        agent.observe(action, next_timestep=timestep)
        for observer in observers:
            observer.observe(env, timestep, action)
        obj_pos_deque.append(np.array(env.push_simulator.getObjPosition()))
        res['episode_return'] += timestep.reward
        res['episode_length'] += 1
        if res['episode_length'] >= max_steps:
            break

    res['steps_per_second'] = res['episode_length'] / (time.time() - start_time)
    for observer in observers:
        res.update(observer.get_metrics())
    #timestep = env.reset() # For video recording of only one episode
    return res

env_config = PoseSubGoalEnvConfig(
# Episode termination config:
    terminate_obj_dist = 12.0,
    goal_dist_tol = 2.0,
    goal_ori_tol= np.pi / 36,
    subgoal_dist_tol = 2.0,
    subgoal_ori_tol= np.pi,
    max_pos_step = 130,
    max_ori_step = np.pi,

    push_simulator_config = PushSimulatorConfig(
        pixels_per_meter=20, width=1024, height=1024,
        obj_proximity_radius=12.0,
        objTuple=(
            # {'name':'Circle', 'radius':4.0},
            {'name': 'Rectangle', 'height': 10.0, 'width': 5.0},
            {'name': 'Polygon', 'vertices': [(5,10), (0,0), (10,0)]},
        ),
        max_dist_obj_goal = 119.99,
        min_dist_obj_goal = 119.9,
        max_ori_obj_goal = np.pi
    )
)

# ENV
def create_environment(ep_i=0):
    env = PoseSubGoalEnv(config=env_config)
    # env = RecordVideoWrapper(env, 'videos', ep_i)
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

    # Load model model
    # agent._learner.restore(savers.restore_from_path('learner_checkpoint'))
    with open(f'learner_checkpoint', 'rb') as f:
        state = pickle.load(f)
        param_holder = ParamHolder(state.params)

    # The actor selects actions according to the policy.
    def policy(params: networks_lib.Params, key: jnp.ndarray,
            observation: jnp.ndarray, epsilon: float) -> jnp.ndarray:
        action_values = network.apply(params, observation)
        return rlax.epsilon_greedy(epsilon).sample(key, action_values)
    actor_core = batched_epsilon_actor_core(policy)
    actor = EGreedyActor(
        actor=actor_core,
        epsilon_start=0.005,
        epsilon_end=0.005,
        epsilon_decay_episodes=1,
        random_key=jax.random.PRNGKey(42),
        variable_client=param_holder,
        adder=None
    )

    observers = [
        SuccessObserver(),
        InitializationObserver()
    ]
    eval_logs_file = open("eval_push_logs_{}.txt".format(exp_num), "a")
    logger = CSVLogger(directory_or_file=eval_logs_file)
    print("Eval Epoch")
    # Common setup:
    eval_eps = 100
    ep_i = 0
    env_config.subgoal_dist_tol = 4.0
    env_config.subgoal_ori_tol = np.pi
    env_config.max_ori_step = np.pi

    for max_pos_step in [130, 60, 30, 15, 7.5, 4.5]:
        env_config.max_pos_step = max_pos_step
        env = create_environment(ep_i=ep_i)
        for i in range(eval_eps):
            eval_res = run_episode(env, actor, observers=observers, max_steps=2000, stuck_steps=10)
            eval_res['subgoal_dist_tol'] = env_config.subgoal_dist_tol
            eval_res['subgoal_ori_tol'] = env_config.subgoal_ori_tol
            eval_res['max_pos_step'] = env_config.max_pos_step
            eval_res['max_ori_step'] = env_config.max_ori_step
            logger.write(eval_res)
            print('Episode: ', ep_i)
            ep_i += 1
            print(eval_res)
        env.reset() # Record video

    eval_logs_file.close()

if __name__ == '__main__':
    # Pass experiment number as argument
    run(sys.argv[1])
    print("End of code")
