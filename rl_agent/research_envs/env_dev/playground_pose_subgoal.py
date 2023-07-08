"""
A playground for testing an environment
"""
# To execute from rl_agent
import sys
sys.path.append('.')

import cv2
import numpy as np

from research_envs.experiment_envs.pose_subgoal_env import PoseSubGoalEnv, PoseSubGoalEnvConfig
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulatorConfig


def key_to_action(key):
    action = -1
    if key == 97: # a
        action = 4
    elif key == 115: # s
        action = 2
    elif key == 100: # d
        action = 0
    elif key  == 119: # w
        action = 6
    return action

if __name__ == "__main__":
    verbose = True
    config = PoseSubGoalEnvConfig(
        push_simulator_config=PushSimulatorConfig(
            max_dist_obj_goal = 30,
            min_dist_obj_goal = 20,
            max_ori_obj_goal = 2*np.pi
        )
    )
    env = PoseSubGoalEnv(config=config)
    env.reset()
    env.render()
    while True:
        # Input handling - requires a cv2 window running => env.render()
        dt = 1.0 / 60.0 #1.0 / 60.0
        key = 0xFF & cv2.waitKey(int(dt * 1000.0)) # Sets default key = 255
        if key == 27: break # Esc key

        action = key_to_action(key)
        if action != -1:
            next_state, reward, done, info = env.step(action)
            # print(next_state['aux_info'])
            if verbose:
                print('Reward: {:.2f} Done: {} Info: {}'.format(reward, done, info))
                print('Dist to obj: {:.2f} Dist to ori: {:.2f}'.format(env.push_simulator.distToObjective(), env.push_simulator.distToOrientation()))
            env.render()

            if done == True:
                env.reset()
                env.render()


