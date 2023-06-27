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

from research_envs.b2PushWorld.PushSimulatorPose import PushSimulator
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer
from research_envs.b2PushWorld.Object import CircleObj, RectangleObj, PolygonalObj

class PoseSubGoalEnv():
    """
    Runs a single experiment.
    Ends when the object is at goal_tolerance to the goal.
    """
    def __init__(self, resetFn=None):
        # the timestep is used to simulate discrete steps through the
        # engine's integrator and it is calculated in seconds
        self.timestep = 1.0 / 60.0

        # velocity and position iterations are used by the constraint solver
        self.vel_iterator = 6
        self.pos_iterator = 2
        
        # Only for drawing
        self.object_distance = 12

        # simulator initialization
        self.push_simulator = PushSimulator(
            pixelsPerMeter=20, width=1024, height=1024, 
            objectiveRadius=2.0, objProxRadius=self.object_distance)

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
        self.reset_fn = resetFn if resetFn is not None else interpolation_reset_fn

    def reset(self):
        self.reset_fn(self)
        self.step_cnt = 0
        # get new observation for a new epoch or simulation
        observation = self.getObservation()
        # observation, info
        return observation  
    
    # def reset(
    #     self,
    #     init_state, 
    #     goal_l, 
    #     goal_tolerance={'pos':2, 'angle': np.pi/36}, 
    #     subgoal_tolerance={'pos':4, 'angle': np.pi/18}, 
    #     obstacle_l=[]
    #     ):
    #     """
    #     Resets the environment to a init_state.
    #     Input:
    #         init_state: {
    #             'obj': research_envs.b2PushWorld.Object,
    #             'obj_pos': (x,y),
    #             'obj_angle': theta,
    #             'robot_pos': (x,y),
    #         },
    #         goal_l: list of {'pos': (x,y), 'angle': theta} - len must be larger than 0.
    #         goal_tolerance: {'pos':float, 'angle':float},
    #         subgoal_tolerance: {'pos':float, 'angle':float},
    #         obstacle_l: list of research_envs.b2PushWorld.Object,
    #     """
    #     assert len(goal_l) > 0, "goal_l must have at least one subgoal."
    #     self.step_cnt = 0
    #     self.push_simulator.reset()
    #     # Intialize object
    #     self.push_simulator.obj = init_state['obj']
    #     self.push_simulator.obj.obj_rigid_body.position = init_state['obj_pos']
    #     self.push_simulator.obj.obj_rigid_body.angle = init_state['obj_angle']
    #     # Initialize goal
    #     self.push_simulator.goal.x = goal_l[0]['pos'][0]
    #     self.push_simulator.goal.y = goal_l[0]['pos'][1]
    #     self.push_simulator.goal_orientation = goal_l[0]['angle']
    #     self.cur_goal_idx = 0
    #     # Initialize robot
    #     self.push_simulator.agent.agent_rigid_body.position = init_state['robot_pos']
        
    #     self.goal_l = goal_l
    #     self.goal_tol = goal_tolerance
    #     self.subgoal_tol = subgoal_tolerance
    #     self.obstacle_l = obstacle_l


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

    def render(self):
        """
        Renders the environment.
        Renders the obstacles separately.
        """
        scene_img = self.push_simulator.drawToBuffer()
        for obj in self.obstacle_l:
            self.obj.Draw(self.push_simulator.pixels_per_meter, scene_img, (0, 0, 1, 0), -1)
        self.scene_buffer.PushFrame(scene_img)
        self.robot_img_state.PushFrame(self.push_simulator.getStateImg())
        self.scene_buffer.Draw()
        self.robot_img_state.Draw()
        cv2.waitKey(1)


def interpolate_scalar(start, end, step_sz):
    # Returns a list of scalars interpolated between start and end.
    # Start can be greater than end.
    # The list length depends on the step size.
    assert step_sz > 0, "step_sz must be greater than 0."
    if start == end:
        return [start]
    if start < end:
        return list(np.arange(start, end+step_sz, step_sz))
    else:
        return list(np.arange(start, end-step_sz, -step_sz))

def interpolation_reset_fn(env):
    """
    Resets the environment
        init_state: {
            'obj': research_envs.b2PushWorld.Object,
            'obj_pos': (x,y),
            'obj_angle': theta,
            'robot_pos': (x,y),
        },
        goal_l: list of {'pos': (x,y), 'angle': theta} - len must be larger than 0.
        goal_tolerance: {'pos':float, 'angle':float},
        subgoal_tolerance: {'pos':float, 'angle':float},
        obstacle_l: list of research_envs.b2PushWorld.Object,
    """
    env.push_simulator.reset() # Randomly resets goal, robot and object

    # Calculate linear interpolation between start and goal.
    x_l = interpolate_scalar(
        env.push_simulator.obj.obj_rigid_body.position[0], env.push_simulator.goal.x, 1)
    y_l = interpolate_scalar(
        env.push_simulator.obj.obj_rigid_body.position[1], env.push_simulator.goal.y, 1)
    angle_l = interpolate_scalar(
        env.push_simulator.obj.obj_rigid_body.angle, env.push_simulator.goal_orientation, np.pi/36)

    max_len = max(max(len(x_l), len(y_l)), len(angle_l))
    goal_l = []
    for i in range(1, max_len):
        goal_l.append({
            'pos': (x_l[min(i, len(x_l)-1)], y_l[min(i, len(y_l)-1)]),
            'angle': angle_l[min(i, len(angle_l)-1)]
        })
    env.goal_l = goal_l

    # Initialize goal with the first subgoal
    env.push_simulator.goal.x = goal_l[0]['pos'][0]
    env.push_simulator.goal.y = goal_l[0]['pos'][1]
    env.push_simulator.goal_orientation = goal_l[0]['angle']
    env.cur_goal_idx = 0
    
    env.goal_tol = {'pos':2, 'angle': np.pi/36}
    env.subgoal_tol = {'pos':4, 'angle': np.pi/9}
    env.obstacle_l = []
    env.checkSuccess()

# def test():
#     environment = PoseSubGoalEnv()
#     init_state = {
#         'obj': PolygonalObj(simulator=environment.push_simulator, x=0.5, y=0.5, vertices=[(5,10), (0,0), (10,0)]),
#         'obj_pos': (15.0, 15.5),
#         'obj_angle': 0.0,
#         'robot_pos': (0.1, 0.1)
#     }
#     obs = environment.reset(init_state=init_state, goal_l=[{'pos':(10.0, 10.0), 'angle':0.0}])
#     reward = 0.0
#     done = False
#     info = {}
#     while True:
#         action = environment.getRandomValidAction()
#         obs, reward, done, info = environment.step(action)
#         environment.render()

#         if done == True:
#             environment.reset(init_state=init_state, goal_l=[{'pos':(10.0, 10.0), 'angle':0.0}])
# if __name__ == '__main__':
#     test()