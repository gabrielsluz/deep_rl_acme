"""
A playground for testing an environment
"""
# To execute from rl_agent
import sys
sys.path.append('.')

import cv2

from research_envs.envs.box2D_img_pushing_pose_env import Box2DPushingEnv, Box2DPushingEnvConfig
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulatorConfig
# from research_envs.envs.box2D_img_pushing_env import Box2DPushingEnv
from research_envs.envs.rewards import RewardFunctions
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
    config = Box2DPushingEnvConfig(
        reward_fn_id=RewardFunctions.PROGRESS,
        max_steps=200,
        push_simulator_config=PushSimulatorConfig(
            objTuple = (
                {'name':'Circle', 'radius':2.0},
                {'name':'Circle', 'radius':4.0},
                {'name':'Circle', 'radius':8.0},
                {'name': 'Rectangle', 'height': 5.0, 'width': 5.0},
                {'name': 'Rectangle', 'height': 10.0, 'width': 5.0},
                {'name': 'Rectangle', 'height': 15.0, 'width': 5.0},
                {'name': 'Polygon', 'vertices': [(5,10), (0,0), (10,0)]},
                {'name': 'Polygon', 'vertices': [(2,10), (0,0), (12,0)]},
                {'name': 'Polygon', 'vertices': [(0,10), (0,0), (10,0)]},
            )
        ),
    )
    env = Box2DPushingEnv(config=config)
    # for obj in env.push_simulator.obj_l:
    #     print(obj.obj_rigid_body.mass, obj.obj_rigid_body.inertia)
    
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
                print('Dist to obj: {:.2f} Dist to ori: {:.2f}'.format(
                    env.push_simulator.distToObjective(), env.push_simulator.distToOrientation()))
                print('Dist to goal: {:.2f}'.format(env.push_simulator.distToObjective()))
            env.render()

            if done == True:
                env.reset()
                env.render()


