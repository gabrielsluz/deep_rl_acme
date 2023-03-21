"""
A playground for testing an environment
"""
# To execute from rl_agent
import sys
sys.path.append('.')

import cv2

from research_envs.envs.box2D_img_pushing_pose_env import Box2DPushingEnv
from research_envs.envs.rewards import RewardFunctions


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
    env = Box2DPushingEnv(smoothDraw=False, reward=RewardFunctions.PROJECTION)
    
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
            if verbose:
                print('Reward: {:.2f} Done: {} Info: {}'.format(reward, done, info))
            env.render()

            if done == True:
                env.reset()
                env.render()


