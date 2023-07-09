""""
Runs a single experiment.
Input:
    - RL policy
    - Obstacle list
    - Object path of subgoals
    - Initial state
    - Goal tolerance


First implementation: Only subgoals.
Ideal => Wrap of the PushSimulatorPose class.
PushSimulator:
reset
loop goal:
    loop subgoal:
        loop update

It will be similar to the env, without rewards and with subgoals.
"""

import numpy as np
import cv2
from gym.spaces import Box, Discrete, Dict
from Box2D import b2Transform

from research_envs.b2PushWorld.PushSimulatorPose import PushSimulator, PushSimulatorConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer


import dataclasses

def interpolate_scalar(start, end, step_sz, min_step_sz):
    # Returns a list of scalars interpolated between start and end.
    # Start can be greater than end.
    # The list length depends on the step size.
    assert step_sz > 0, "step_sz must be greater than 0."
    if start == end:
        return [start]
    if start < end:
        l = list(np.arange(start, end, step_sz))
    else:
        l = list(np.arange(start, end, -step_sz))
    if abs(end - l[-1]) > min_step_sz:
        return l + [end]
    else:
        return l[:-1] + [end]

def interpolation_reset_fn(env):
    """
    Resets the environment and generates subgoals by linear interpolation.
        init_state: {
            'obj': research_envs.b2PushWorld.Object,
            'obj_pos': (x,y),
            'obj_angle': theta,
            'robot_pos': (x,y),
        },
        goal_l: list of {'pos': (x,y), 'angle': theta} - len must be larger than 0.
        goal_tolerance: {'pos':float, 'angle':float},
        # subgoal_tolerance: {'pos':float, 'angle':float},
        obstacle_l: list of research_envs.b2PushWorld.Object,
    """
    env.push_simulator.reset() # Randomly resets goal, robot and object

    # Calculate linear interpolation between start and goal.
    start_c = [
        env.push_simulator.obj.obj_rigid_body.position[0], 
        env.push_simulator.obj.obj_rigid_body.position[1], 
        env.push_simulator.obj.obj_rigid_body.angle]
    goal_c = [   
        env.push_simulator.goal.x,
        env.push_simulator.goal.y,
        env.push_simulator.goal_orientation]
    # Adjust angle values so that interpolation takes the shortest path
    start_c[2] = start_c[2] % (2*np.pi)
    goal_c[2] = goal_c[2] % (2*np.pi)
    # Find the shortest path
    big_angle = max(start_c[2], goal_c[2])
    small_angle = min(start_c[2], goal_c[2])
    clockwise_dist = big_angle - small_angle
    counterclockwise_dist = (2*np.pi - big_angle) + small_angle
    if counterclockwise_dist < clockwise_dist:
        # small_angle = small_angle + 2*np.pi
        if start_c[2] <= goal_c[2]:
            start_c[2] = start_c[2] + 2*np.pi
        else:
            goal_c[2] = goal_c[2] + 2*np.pi
    # Calculate interpolation
    max_pos_step = env.max_pos_step
    max_ori_step = env.max_ori_step
    pos_dist = np.linalg.norm(np.array(start_c[:2]) - np.array(goal_c[:2]))
    ori_diff = abs(start_c[2] - goal_c[2])
    num_steps = max(ori_diff/max_ori_step, pos_dist/max_pos_step)
    num_steps = int(np.ceil(num_steps))+1
    interp_arr = np.linspace(
            start_c,
            goal_c,
            num=num_steps,
            endpoint=True
    )
    goal_l = []
    for i in range(1, num_steps):
        goal_l.append({
            'pos': (interp_arr[i,0], interp_arr[i,1]),
            'angle': interp_arr[i,2]
        })
    env.goal_l = goal_l 

    # Initialize goal with the first subgoal
    env.push_simulator.goal.x = goal_l[0]['pos'][0]
    env.push_simulator.goal.y = goal_l[0]['pos'][1]
    env.push_simulator.goal_orientation = goal_l[0]['angle']
    env.cur_goal_idx = 0
    
    env.goal_tol = {'pos':env.goal_dist_tol, 'angle': env.goal_ori_tol}
    # env.subgoal_tol = {'pos':4, 'angle': np.pi/9}
    env.obstacle_l = []
    env.checkSuccess()

@dataclasses.dataclass
class PoseSubGoalEnvConfig:
    """Configuration options for the PoseSubGoalEnv.
    Attributes:
        terminate_obj_dist: If the robot is further than this distance from the object
            the episode terminates.
        goal_dist_tol: Necessary distance to consider the goal reached.
        goal_ori_tol: Necessary orientation to consider the goal reached.
        reset_fn: Function that receives the env as input and resets its internal state.
            The idea is to make significant changes to env every reset.

        push_simulator_config: Configuration options for the PushSimulator.
    """
    # Episode termination config:
    terminate_obj_dist: float = 14.0
    goal_dist_tol: float = 2.0
    goal_ori_tol: float = np.pi / 36
    max_pos_step: float = 30
    max_ori_step: float = np.pi/2
    reset_fn: callable = interpolation_reset_fn

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


class PoseSubGoalEnv():
    """
    Runs a single experiment.
    Ends when the object is at goal_tolerance to the goal.
    """
    def __init__(self, config=PoseSubGoalEnvConfig):
        # the timestep is used to simulate discrete steps through the
        # engine's integrator and it is calculated in seconds
        self.timestep = 1.0 / 60.0

        # velocity and position iterations are used by the constraint solver
        self.vel_iterator = 6
        self.pos_iterator = 2
        
        # Only for drawing
        self.object_distance = config.terminate_obj_dist

        # simulator initialization
        self.push_simulator = PushSimulator(config.push_simulator_config)

        # keep track of this environment state shape for outer references
        self.state_shape = self.push_simulator.state_shape
        self.observation_space = Dict({	
            'state_img': Box(low=0.0, high=1.0, shape=self.state_shape, dtype=np.float32),	
            'aux_info': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)})
        self.action_space = Discrete(self.push_simulator.agent.directions)
        self.max_objective_dist = self.push_simulator.max_dist_obj_goal * 1.2

        self.scene_buffer = CvDrawBuffer(window_name="Push Simulation", resolution=(1024,1024))
        self.robot_img_state = CvDrawBuffer(window_name="Image State", resolution=(16,16))

        self.step_cnt = 0

        # reset
        self.reset_fn = config.reset_fn
        self.max_pos_step = config.max_pos_step
        self.max_ori_step = config.max_ori_step
        self.goal_dist_tol = config.goal_dist_tol
        self.goal_ori_tol = config.goal_ori_tol
        self.reset()

    def reset(self):
        self.reset_fn(self)
        self.step_cnt = 0
        # get new observation for a new epoch or simulation
        observation = self.getObservation()
        # observation, info
        return observation  

    def checkSuccess(self):
        """
         Checks if the object has reached the goal.
         Updates the subgoal if necessary.
        """
        # Checks if the object is in the safe zone and in the correct orientation +- epsilon
        dist_to_objetive = self.push_simulator.distToObjective()
        dist_to_orientation = abs(self.push_simulator.distToOrientation())
        reached_sub = dist_to_objetive < self.goal_tol['pos'] and dist_to_orientation < self.goal_tol['angle']
        if self.cur_goal_idx == len(self.goal_l)-1: # Last goal
            return reached_sub
        reached_final = False
        if reached_sub: # Reached subgoal
            self.cur_goal_idx += 1
            self.push_simulator.goal.x = self.goal_l[self.cur_goal_idx]['pos'][0]
            self.push_simulator.goal.y = self.goal_l[self.cur_goal_idx]['pos'][1]
            self.push_simulator.goal_orientation = self.goal_l[self.cur_goal_idx]['angle']
            reached_final = self.checkSuccess() # Recursively check if the new subgoal is reached
        return reached_final

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
        while(self.push_simulator.agent.IsPerformingAction()):
            self.push_simulator.update(
                timeStep=self.timestep, velocity_iterator=self.vel_iterator, position_iterator=self.pos_iterator)
            # Check for collisions

        # get the last state for safe computation
        observation = self.getObservation()

        if self.checkSuccess():
            done = True
            info = {'success': True}

        return observation, reward, done, info
    
    def getRandomValidAction(self):
        return self.push_simulator.agent.GetRandomValidAction()


    def drawObjInSubGoal(self, image, position, angle, color=(0.0, 1.0, 0.0)):
        # Draw the object in the goal location with the goal orientation
        obj = self.push_simulator.obj
        if obj.obj_type == 'Circle':
            screen_pos = self.push_simulator.worldToScreen((position[0], position[1]))
            cv2.circle(
                image, screen_pos, int(obj.obj_radius*self.push_simulator.pixels_per_meter),color,
                thickness=4, lineType=4)
            return
        # Rotate and Translate
        transform_matrix = b2Transform()
        transform_matrix.SetIdentity()
        transform_matrix.Set(position, angle)
        vertices = [transform_matrix * v for v in obj.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.push_simulator.worldToScreen(v) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)
        self.push_simulator.drawArrow(image, position, angle, 10, color)

    def render(self):
        """
        Renders the environment.
        Renders the obstacles separately.
        """
        scene_img = self.push_simulator.drawToBuffer()
        for obj in self.obstacle_l:
            self.obj.Draw(self.push_simulator.pixels_per_meter, scene_img, (0, 0, 1, 0), -1)
        for subgoal in self.goal_l[self.cur_goal_idx+1:]:
            self.drawObjInSubGoal(scene_img, subgoal['pos'], subgoal['angle'], (0.0, 1.0, 0.0))

        self.scene_buffer.PushFrame(scene_img)
        self.robot_img_state.PushFrame(self.push_simulator.getStateImg())
        self.scene_buffer.Draw()
        self.robot_img_state.Draw()
        cv2.waitKey(1)