import sys
sys.path.append('../..')

import numpy as np

from acme.wrappers import GymWrapper
# from wrappers.add_channel_wrapper import AddChannelDimWrapper
from wrappers.frame_stack_wrapper import FrameStackWrapper
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
    env = GymWrapper(env)
    env = FrameStackWrapper(env, frameStackDepth=4)
    return env

def main():
    env = create_environment()
    env.reset()

if __name__ == '__main__':
    main()