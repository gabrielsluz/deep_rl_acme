"""
The goal is a position + orientation => pose

Modifications:
- Done condition - Done
- Random orientation start - Done
- Robot state - Done => check a little more
- Reward function - Done

Not working well, hypothesis to improve the performance:
- Remove projection reward when object is in goal position
- Improve reward function
- Add distance from object to objective in the state
- Use continous actions
- Use a different algorithm

"""

import numpy as np
import cv2
from threading import Thread
from gym.spaces import Box, Discrete, Dict

from research_envs.envs.rewards import RewardFunctions
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulator
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

class Box2DPushingEnv():
    def __init__(self, smoothDraw=True, reward=RewardFunctions.PROJECTION, max_steps=100, d=30):
        print('Box2d Pushing Environment with pose goal')
        # the timestep is used to simulate discrete steps through the
        # engine's integrator and it is calculated in seconds
        self.timestep = 1.0 / 60.0

        # velocity and position iterations are used by the constraint solver
        self.vel_iterator = 6
        self.pos_iterator = 2
        
        # restrictions 
        self.object_distance = 12
        self.safe_zone_radius = 2
        self.orientation_eps = 0.174533 *2 # 10 degrees

        # simulator initialization
        self.push_simulator = PushSimulator(
            pixelsPerMeter=20, width=1024, height=1024, 
            objectiveRadius=self.safe_zone_radius, objProxRadius=self.object_distance, d=d)
        self.smooth_draw = smoothDraw

        # keep track of this environment state shape for outer references
        self.state_shape = self.push_simulator.state_shape
        self.observation_space = Dict({	
            'state_img': Box(low=0.0, high=1.0, shape=self.state_shape, dtype=np.float32),	
            'aux_info': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)})
        self.action_space = Discrete(self.push_simulator.agent.directions)
        self.max_objective_dist = np.sqrt(	
            (self.push_simulator.width/self.push_simulator.pixels_per_meter)** 2 + 	
            (self.push_simulator.height/self.push_simulator.pixels_per_meter)** 2)	


        # async cv buffers for full step rasterization
        # without frameskip
        self.scene_buffer = CvDrawBuffer(window_name="Push Simulation", resolution=(1024,1024))
        # self.robot_img_state = CvDrawBuffer(window_name="Image State", resolution=(320,320))
        self.robot_img_state = CvDrawBuffer(window_name="Image State", resolution=(16,16))

        if self.smooth_draw == True:
            self.draw_thread = Thread(target=self.threadedRendering)
            self.draw_thread.start()

        self.reward_func = reward
        # End episode after max_steps
        self.step_cnt = 0
        self.max_steps = max_steps

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

            if self.smooth_draw == True:
                # performing sampling if delta is higher than frequency in ms
                #if sampling >= self.sampling_frequency and self.smooth_draw == True:
                async_buffer = self.push_simulator.drawToBuffer()
                img_state = self.push_simulator.getStateImg()
                self.scene_buffer.PushFrame(async_buffer)
                self.robot_img_state.PushFrame(img_state)

        # get the last state for safe computation
        observation = self.getObservation()

        # if smooth draw is off, get 
        # the framebuffer only after performing
        # a full step 
        if self.smooth_draw == False:
            async_buffer = self.push_simulator.drawToBuffer()
            img_state = self.push_simulator.getStateImg()
            self.scene_buffer.PushFrame(async_buffer)
            self.robot_img_state.PushFrame(img_state)

        # check if agent broke restriction 
        dist_to_object = self.push_simulator.distToObject()
        if dist_to_object > self.object_distance:
            done = True

        if self.checkSuccess():
            done = True
            info = {'success': True}

        # compute reward
        # if self.reward_func == RewardFunctions.FOCAL_POINTS:
            # reward = self.rewardFocalPoints()
        # if self.reward_func == RewardFunctions.REACHING_PROJECTION:
            # reward = self.rewardReachingProjection()
        if self.reward_func == RewardFunctions.PROJECTION:            
            reward = self.rewardProjection()
        if self.reward_func == RewardFunctions.PROGRESS:            
            reward = self.rewardProgress()

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

        # Fill render
        async_buffer = self.push_simulator.drawToBuffer()
        self.scene_buffer.PushFrame(async_buffer)
        self.robot_img_state.PushFrame(observation['state_img'])

        # observation, info
        return observation

    def threadedRendering(self):
        while True:
            self.scene_buffer.Draw()
            self.robot_img_state.Draw()
            cv2.waitKey(1)      

    def render(self):
        # prevent draw calls if async thread is running
        if self.smooth_draw == True:
            return

        self.scene_buffer.Draw()
        self.robot_img_state.Draw()
        cv2.waitKey(1)

    def close(self):
        # TODO
        return

def test():
    # when using the environment explicitly you do not need to
    # register it into the GyM API inside the __init__.py of your
    # custom GyM environment.
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