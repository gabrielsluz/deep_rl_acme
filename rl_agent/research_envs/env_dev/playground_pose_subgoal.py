"""
A playground for testing an environment
"""
# To execute from rl_agent
import sys
sys.path.append('.')

import cv2

from research_envs.experiment_envs.pose_subgoal_env import PoseSubGoalEnv, PoseSubGoalEnvConfig
# from research_envs.envs.box2D_img_pushing_env import Box2DPushingEnv
from research_envs.b2PushWorld.Object import CircleObj, RectangleObj, PolygonalObj


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
    env = PoseSubGoalEnv()
    # init_state = {
    #     'obj': PolygonalObj(simulator=env.push_simulator, x=0.5, y=0.5, vertices=[(5,10), (0,0), (10,0)]),
    #     'obj_pos': (15.0, 15.5),
    #     'obj_angle': 0.0,
    #     'robot_pos': (0.1, 0.1)
    # }
    # goal_l = [
    #     {'pos':(10.0, 10.0), 'angle':0.0},
    #     {'pos':(40.0, 40.0), 'angle':0.0}

    # ]
    # env.reset(init_state=init_state, goal_l=goal_l)
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


