import numpy as np
import cv2
from gym.spaces import Box, Discrete, Dict

from research_envs.envs.rewards import RewardFunctions
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulator, PushSimulatorConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import dataclasses

@dataclasses.dataclass
class Box2DPushingEnvConfig:
    """Configuration options for the Box2DPushingEnv.
    Attributes:
        terminate_obj_dist: If the robot is further than this distance from the object
            the episode terminates.
        goal_dist_tol: Necessary distance to consider the goal reached.
        goal_ori_tol: Necessary orientation to consider the goal reached.
        max_steps: Maximum number of steps per episode.
        reward_fn_id: Reward function to use, from rewards.py

    """
    # Episode termination config:
    terminate_obj_dist: float = 14.0
    goal_dist_tol: float = 2.0
    goal_ori_tol: float = np.pi / 36
    max_steps: int = 200
    # Reward config:
    reward_fn_id: RewardFunctions = RewardFunctions.PROGRESS
    # Push simulator config:
    push_simulator_config: PushSimulatorConfig = PushSimulatorConfig(
        pixels_per_meter=20, width=1024, height=1024,
        obj_proximity_radius=terminate_obj_dist,
        objTuple=(
            {'name':'Circle', 'radius':4.0},
            {'name': 'Rectangle', 'height': 10.0, 'width': 5.0},
            {'name': 'Polygon', 'vertices': [(5,10), (0,0), (10,0)]},
        ),
        max_dist_obj_goal = 30,
        min_dist_obj_goal = 2,
        max_ori_obj_goal = np.pi / 2
    )


class Box2DPushingEnv():
    def __init__(self, config=Box2DPushingEnvConfig):
        print('Box2d Pushing Environment with pose goal')
        # the timestep is used to simulate discrete steps through the
        # engine's integrator and it is calculated in seconds
        self.timestep = 1.0 / 60.0

        # velocity and position iterations are used by the constraint solver
        self.vel_iterator = 6
        self.pos_iterator = 2
        
        # restrictions 
        self.object_distance = config.terminate_obj_dist
        self.safe_zone_radius = config.goal_dist_tol
        self.orientation_eps = config.goal_ori_tol

        # simulator initialization
        self.push_simulator = PushSimulator(config.push_simulator_config)

        # keep track of this environment state shape for outer references
        self.state_shape = self.push_simulator.state_shape
        self.observation_space = Dict({	
            'state_img': Box(low=0.0, high=1.0, shape=self.state_shape, dtype=np.float32),	
            'aux_info': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)})
        self.action_space = Discrete(self.push_simulator.agent.directions)
        self.max_objective_dist = self.push_simulator.max_dist_obj_goal * 1.2

        # buffers for rendering
        self.scene_buffer = CvDrawBuffer(window_name="Push Simulation", resolution=(1024,1024))
        self.robot_img_state = CvDrawBuffer(window_name="Image State", resolution=(16,16))

        self.reward_func = config.reward_fn_id
        # End episode after max_steps
        self.step_cnt = 0
        self.max_steps = config.max_steps

    def checkSuccess(self):
        # Checks if the object is in the safe zone and in the correct orientation +- epsilon
        dist_to_objetive = self.push_simulator.distToObjective()
        dist_to_orientation = abs(self.push_simulator.distToOrientation())
        return dist_to_objetive < self.safe_zone_radius and dist_to_orientation < self.orientation_eps

    def normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm != 0.0:
            vec = vec / norm
        return vec, norm

    def computeDirectionCoef(self, v1, v2, t=0.2):
        if t < 0.0:
            t = 0.0
        if t >= 0.5:
            t = 0.5

        proj_reward = (1.0 + np.dot(v1, v2)) / (1 + t)
        if proj_reward <= 1.0:
            proj_reward = -(1.0 - proj_reward)
        else:
            proj_reward = proj_reward - 1.0

        return proj_reward


    def rewardProjection(self):
        #Reward computation for the box2D push environment
        total_reward = 0.0
        proj_reward = 0.0
        orient_reward = 0.0
        time_penalty = -0.1

        # projection reward if object not in safe zone
        if self.push_simulator.distToObjective() > self.safe_zone_radius:
            bd, _ = self.normalize(self.push_simulator.getObjPosition() - self.push_simulator.getLastObjPosition())
            bo, _ = self.normalize(self.push_simulator.goal - self.push_simulator.getLastObjPosition())
            proj_reward = self.computeDirectionCoef(bd,bo)

        # Orientation reward in the format of projection reward but comparing current orientation error to previous
        # x_t-1 - x_t
        orient_reward = abs(self.last_orient_error) - abs(self.push_simulator.distToOrientation()/np.pi)

        dist_to_object = self.push_simulator.distToObject()
        if self.checkSuccess():
            return 1.0
        if dist_to_object > self.object_distance:
            return -1.0

        # compute total reward as the 'expected value'
        total_reward = proj_reward * 1.0 + time_penalty * 0.1 + orient_reward * 1.0
        total_reward = total_reward / 2.0

        return total_reward

    def rewardProgress(self):	
        # Reward based on the progress of the agent towards the goal	
        # Limits the maximum reward to [-1.0, 1.0] (except for success or death)	
        total_reward = 0.0	
        progress_reward = 0.0
        orient_reward = 0.0	
        success_reward = 2.0	
        death_penalty = -1.0	
        time_penalty = -0.01	

        dist_to_object = self.push_simulator.distToObject()	
        if self.checkSuccess():
            return success_reward	
        if dist_to_object > self.object_distance:	
            return death_penalty	
        # progress reward	
        last_dist = (self.push_simulator.goal - self.push_simulator.getLastObjPosition()).length	
        cur_dist = (self.push_simulator.goal - self.push_simulator.getObjPosition()).length	
        # Tries to scale between -1 and +1, but also clips it	
        max_gain = 2.0 # Heuristic, should be adapted to the environment	
        progress_reward = (last_dist - cur_dist) / max_gain  	
        progress_reward = max(min(progress_reward, 1.0), -1.0)	

        orient_reward = abs(self.last_orient_error) - abs(self.push_simulator.distToOrientation()/np.pi)
        	
        # compute total reward, weigthing to give more importance to success or death	
        total_reward = progress_reward*0.5 + orient_reward*0.5 + time_penalty	
        return total_reward

    def rewardProgressShaping(self):
        # Two parts: position; position and orientation
        # Reward based on the progress of the agent towards the goal	
        # Limits the maximum reward to [-1.0, 1.0] (except for success or death)	
        total_reward = 0.0	
        progress_reward = 0.0
        orient_reward = 0.0	
        success_reward = 2.0	
        death_penalty = -1.0	
        time_penalty = -0.01	

        dist_to_object = self.push_simulator.distToObject()	
        if self.checkSuccess():
            return success_reward	
        if dist_to_object > self.object_distance:	
            return death_penalty	
        # progress reward	
        last_dist = (self.push_simulator.goal - self.push_simulator.getLastObjPosition()).length	
        cur_dist = (self.push_simulator.goal - self.push_simulator.getObjPosition()).length	
        # Tries to scale between -1 and +1, but also clips it	
        max_gain = 2.0 # Heuristic, should be adapted to the environment	
        progress_reward = (last_dist - cur_dist) / max_gain  	
        progress_reward = max(min(progress_reward, 1.0), -1.0)	

        # If the object is close enough to the goal, we also reward the orientation
        orient_reward = 0.0
        if cur_dist < 8:
            orient_reward = abs(self.last_orient_error) - abs(self.push_simulator.distToOrientation()/np.pi)
        	
        # compute total reward, weigthing to give more importance to success or death	
        total_reward = progress_reward*0.5 + orient_reward*0.5 + time_penalty	
        return total_reward

    def getRandomValidAction(self):
        return self.push_simulator.agent.GetRandomValidAction()

    def getObservation(self):	
        obs = {}	
        obs['state_img'] = self.push_simulator.getStateImg()	
        obs['aux_info'] = np.zeros(shape=(2,), dtype=np.float32)	
        # obs['aux_info'][0] = self.push_simulator.distToObject() / self.push_simulator.obj_proximity_radius	
        # obs['aux_info'][1] = self.push_simulator.distToObjective() / self.max_objective_dist	
        # obs['aux_info'][2] = self.push_simulator.distToOrientation() / np.pi	
        obs['aux_info'][0] = self.push_simulator.distToObjective() / self.max_objective_dist	
        obs['aux_info'][1] = self.push_simulator.distToOrientation() / np.pi	
        return obs

    def step(self, action):
        # return variables
        observation      = {}
        reward           = 0.0
        done             = False
        info             = {'success': False, 'TimeLimit.truncated': False}

        self.push_simulator.agent.PerformAction(action)

        # set previous state for reward calculation
        self.push_simulator.agent.UpdateLastPos()
        self.push_simulator.obj.UpdateLastPos()
        self.last_orient_error = self.push_simulator.distToOrientation()/ np.pi

        # wait until the agent has ended performing its step
        # push draw buffers to avoid raster gaps
        while(self.push_simulator.agent.IsPerformingAction()):
            self.push_simulator.update(timeStep=self.timestep, velocity_iterator=self.vel_iterator, position_iterator=self.pos_iterator)

        # get the last state for safe computation
        observation = self.getObservation()

        # check if agent broke restriction 
        dist_to_object = self.push_simulator.distToObject()
        if dist_to_object > self.object_distance:
            done = True

        if self.checkSuccess():
            done = True
            info = {'success': True}

        # compute reward
        if self.reward_func == RewardFunctions.PROJECTION:            
            reward = self.rewardProjection()
        if self.reward_func == RewardFunctions.PROGRESS:            
            reward = self.rewardProgress()
        if self.reward_func == RewardFunctions.PROGRESS_SHAPING:
            reward = self.rewardProgressShaping()
        

        # Check if time limit exceeded
        self.step_cnt += 1
        if self.step_cnt >= self.max_steps and not info['success']:
            done = True
            info['TimeLimit.truncated'] = True

        return observation, reward, done, info

    def reset(self):
        self.push_simulator.reset()
        self.step_cnt = 0
        # get new observation for a new epoch or simulation
        observation = self.getObservation()
        # observation, info
        return observation   

    def render(self):
        # Draw
        self.scene_buffer.PushFrame(self.push_simulator.drawToBuffer())
        self.robot_img_state.PushFrame(self.push_simulator.getStateImg())
        self.scene_buffer.Draw()
        self.robot_img_state.Draw()
        cv2.waitKey(1)

    def close(self):
        # TODO
        return

def test():
    environment = Box2DPushingEnv(smoothDraw=False)
    obs = environment.reset()
    reward = 0.0
    done = False
    info = {}
    while True:
        action = environment.getRandomValidAction()
        obs, reward, done, info = environment.step(action)
        environment.render()

        if done == True:
            environment.reset()