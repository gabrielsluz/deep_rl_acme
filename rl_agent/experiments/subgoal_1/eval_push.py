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
from research_envs.experiment_envs.pose_subgoal_env import PoseSubGoalEnv
from observers.success_observer import SuccessObserver
from acme.utils.loggers import CSVLogger

# Utils
# def calc_suc_rate(data: list) -> float:
#     suc_cnt = 0
#     for i in data:
#         suc_cnt += i['success']
#     return suc_cnt / len(data)

class ParamHolder():
    def __init__(self, params):
        self.params = params

# Eval loop
# episode_length,episode_return,episodes,steps,steps_per_second,success
def run_episode(env, agent, observers=[], max_steps=200):
    res = {
        'episode_length': 0,
        'episode_return': 0,
    }
    start_time = time.time()
    timestep = env.reset()
    agent.observe_first(timestep)
    for observer in observers:
      observer.observe_first(env, timestep)

    while not timestep.last():
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)

        agent.observe(action, next_timestep=timestep)
        for observer in observers:
            observer.observe(env, timestep, action)
        res['episode_return'] += timestep.reward
        res['episode_length'] += 1
        if res['episode_length'] >= max_steps:
            break

    res['steps_per_second'] = res['episode_length'] / (time.time() - start_time)
    for observer in observers:
        res.update(observer.get_metrics())
    # timestep = env.reset() # For video recording of only one episode
    return res


# ENV
def create_environment():
    env = PoseSubGoalEnv()
    env = RecordVideoWrapper(env, 'videos')
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
    with open(f'learner_checkpoint_{exp_num}', 'rb') as f:
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
        SuccessObserver()
    ]
    eval_eps = 100

    eval_logs_file = open("eval_push_logs_{}.txt".format(exp_num), "a")
    logger = CSVLogger(directory_or_file=eval_logs_file)
    print("Eval Epoch")
    for i in range(eval_eps):
        eval_res = run_episode(env, actor, observers=observers)
        logger.write(eval_res)
        print(eval_res)
    # suc_rate = calc_suc_rate(loop._logger.data[-eval_eps:])
    # print('EVAL in {} episodes: Success Rate: {:.3f}'.format(eval_eps, suc_rate))
    eval_logs_file.close()

if __name__ == '__main__':
    # Pass experiment number as argument
    run(sys.argv[1])
    print("End of code")


"""
O que fazer?
- Loop de episódio => max steps
- Agregar métricas

"""