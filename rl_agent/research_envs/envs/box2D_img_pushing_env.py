import numpy as np
import cv2
from threading import Thread
from gym.spaces import Box, Discrete

from research_envs.envs.rewards import RewardFunctions
from research_envs.b2PushWorld.PushSimulator import PushSimulator
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

class Box2DPushingEnv():
    def __init__(self, smoothDraw=True, reward=RewardFunctions.FOCAL_POINTS, max_steps=100, d=30):
        # the timestep is used to simulate discrete steps through the
        # engine's integrator and it is calculated in seconds
        self.timestep = 1.0 / 60.0

        # velocity and position iterations are used by the constraint solver
        self.vel_iterator = 6
        self.pos_iterator = 2
        
        # restrictions 
        self.object_distance = 12
        self.safe_zone_radius = 5

        # simulator initialization
        self.push_simulator = PushSimulator(
            pixelsPerMeter=20, width=1024, height=1024, 
            objectiveRadius=self.safe_zone_radius, objProxRadius=self.object_distance, d=d)
        self.smooth_draw = smoothDraw

        # keep track of this environment state shape for outer references
        self.state_shape = self.push_simulator.state_shape
        self.observation_space = Box(low=0.0, high=1.0, shape=self.state_shape[:2], dtype=np.float32)
        self.action_space = Discrete(self.push_simulator.agent.directions)

        # async cv buffers for full step rasterization
        # without frameskip
        self.scene_buffer = CvDrawBuffer(window_name="Push Simulation", resolution=(1024,1024))
        self.robot_img_state = CvDrawBuffer(window_name="Image State", resolution=(320,320))

        if self.smooth_draw == True:
            self.draw_thread = Thread(target=self.threadedRendering)
            self.draw_thread.start()

        self.reward_func = reward
        # End epsisode after max_steps
        self.step_cnt = 0
        self.max_steps = max_steps

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

    def rewardFocalPoints(self):
        #Reward computation for the box2D push environment
        reward = 0.0
        time_penalty = -0.1

        # projection reward
        bd, _ = self.normalize(self.push_simulator.getObjPosition() - self.push_simulator.getLastObjPosition())
        bo, _ = self.normalize(self.push_simulator.goal - self.push_simulator.getLastObjPosition())

        # push points
        agent_last_pos = self.push_simulator.getLastAgentPosition()
        obj_last_pos = self.push_simulator.getLastObjPosition()
        objective_pos = self.push_simulator.goal
        agent_move, agent_move_norm = self.normalize(self.push_simulator.getAgentPosition() - agent_last_pos)
        objective_to_agent, d1 = self.normalize(agent_last_pos - objective_pos)
        objective_to_obj, d2 = self.normalize(obj_last_pos - objective_pos)
        left = np.array([-objective_to_obj.y, objective_to_obj.x])
        right = np.array([objective_to_obj.y, -objective_to_obj.x])
        p1 = obj_last_pos + left * 8.0
        p2 = obj_last_pos + right * 8.0
        h1, h1_norm = self.normalize(p1 - agent_last_pos)
        h2, h2_norm = self.normalize(p2 - agent_last_pos)
        
        if d1 < d2:
            if h1_norm < h2_norm:
                reward = self.computeDirectionCoef(agent_move, h1)
            else:
                reward = self.computeDirectionCoef(agent_move, h2)
        else:
            reward = self.computeDirectionCoef(bd, bo)
        
        dist_to_objetive = self.push_simulator.distToObjective()
        dist_to_object = self.push_simulator.distToObject()
        if dist_to_objetive < self.safe_zone_radius:
            return 1.0
        if dist_to_object > self.object_distance:
            return -1.0

        # compute total reward as the 'expected value'
        total_reward = reward * 1.0 + time_penalty * 0.1
        total_reward = total_reward / 2.0

        return total_reward

    def rewardReachingProjection(self):
        #Reward computation for the box2D push environment
        total_reward = 0.0
        proj_reward = 0.0
        reaching_reward = 0.0
        time_penalty = -0.1

        # projection reward
        bd, _ = self.normalize(self.push_simulator.getObjPosition() - self.push_simulator.getLastObjPosition())
        bo, _ = self.normalize(self.push_simulator.goal - self.push_simulator.getLastObjPosition())

        where_it_should, _ = self.normalize(self.push_simulator.getObjPosition() - self.push_simulator.getLastAgentPosition())
        where_it_went, _ = self.normalize(self.push_simulator.getAgentPosition() - self.push_simulator.getLastAgentPosition())

        proj_reward = self.computeDirectionCoef(bd,bo)
        reaching_reward = self.computeDirectionCoef(where_it_should,where_it_went)

        dist_to_objetive = self.push_simulator.distToObjective()
        dist_to_object = self.push_simulator.distToObject()
        if dist_to_objetive < self.safe_zone_radius:
            return 1.0
        if dist_to_object > self.object_distance:
            return -1.0

        # compute total reward as the 'expected value'
        total_reward = reaching_reward * 0.1 + proj_reward * 1.0 + time_penalty * 0.1
        total_reward = total_reward / 3.0

        return total_reward

    def rewardProjection(self):
        #Reward computation for the box2D push environment
        total_reward = 0.0
        proj_reward = 0.0
        time_penalty = -0.1

        # projection reward
        bd, _ = self.normalize(self.push_simulator.getObjPosition() - self.push_simulator.getLastObjPosition())
        bo, _ = self.normalize(self.push_simulator.goal - self.push_simulator.getLastObjPosition())
        proj_reward = self.computeDirectionCoef(bd,bo)

        dist_to_objetive = self.push_simulator.distToObjective()
        dist_to_object = self.push_simulator.distToObject()
        if dist_to_objetive < self.safe_zone_radius:
            return 1.0
        if dist_to_object > self.object_distance:
            return -1.0

        # compute total reward as the 'expected value'
        total_reward = proj_reward * 1.0 + time_penalty * 0.1
        total_reward = total_reward / 2.0

        return total_reward

    def getRandomValidAction(self):
        return self.push_simulator.agent.GetRandomValidAction()

    def step(self, action):
        # return variables
        observation      = []
        reward           = 0.0
        done             = False
        info             = {'success': False, 'TimeLimit.truncated': False}

        self.push_simulator.agent.PerformAction(action)

        # set previous state for reward calculation
        self.push_simulator.agent.UpdateLastPos()
        self.push_simulator.obj.UpdateLastPos()

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
        observation = self.push_simulator.getStateImg()

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

        # check if object reached objective zone
        dist_to_objetive = self.push_simulator.distToObjective()
        if dist_to_objetive < self.safe_zone_radius:
            done = True
            info = {'success': True}

        # compute reward
        if self.reward_func == RewardFunctions.FOCAL_POINTS:
            reward = self.rewardFocalPoints()
        if self.reward_func == RewardFunctions.PROJECTION:            
            reward = self.rewardProjection()
        if self.reward_func == RewardFunctions.REACHING_PROJECTION:
            reward = self.rewardReachingProjection()

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
        observation = self.push_simulator.getStateImg()

        # Fill render
        async_buffer = self.push_simulator.drawToBuffer()
        self.scene_buffer.PushFrame(async_buffer)
        self.robot_img_state.PushFrame(observation)

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