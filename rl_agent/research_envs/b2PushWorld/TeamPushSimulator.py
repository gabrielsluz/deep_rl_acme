from random import randint
import random
import cv2
import numpy as np
from Box2D import b2World, b2Vec2
from research_envs.b2PushWorld.Agent import Agent
from research_envs.b2PushWorld.Object import Object

"""
Push Simulator with a single object and multiple agents.
"""

class TeamPushSimulator:
    def __init__(self, pixelsPerMeter = 20, width = 1024, height = 1024, objectiveRadius = 3.0, objProxRadius=15, d=30, n_agents=2):
        # py game is used for simple rendering to avoid
        # box2D framework
        self.pixels_per_meter = pixelsPerMeter
        self.width = width
        self.height = height
        self.screen = np.zeros(shape=(self.height, self.width), dtype=np.float32)
        self.state_shape = (16,16,1)
        self.observed_dist_shape = (320,320,3)

        # ----------- world creation ----------
        # crate a simple box2D world with -9.8 gravity acceleration
        self.world = b2World(gravity=(0, 0.0), doSleep=False)

        # ----------- specify goal position -----------
        self.goal_radius = objectiveRadius
        self.obj_proximity_radius = objProxRadius
        self.goal = b2Vec2(0,0)

        # ----------- specify agent and pushing object -----------
        self.obj   = Object(simulator=self, x=25, y=25, radius=4.0)
        self.n_agents = n_agents
        self.agents = []
        for _ in range(n_agents):
            self.agents.append(
                Agent(simulator=self, x=30, y=25, radius=1.0, velocity=1.0, forceLength=2.0, totalDirections=8))

        # Define object - goal - robot reset distances
        self.d = d

    def reset(self):
        # Generates the box position
        max_x = int(self.width/self.pixels_per_meter)
        max_y = int(self.height/self.pixels_per_meter)
        box_pos = [random.uniform(0, max_x), random.uniform(0, max_y)]

        self.obj.obj_rigid_body.position = (box_pos[0], box_pos[1])

        # Generates the goal position at a distance d of the box.
        rand_rad = random.uniform(-2*np.pi, 2*np.pi)
        rand_dir = [np.cos(rand_rad), np.sin(rand_rad)]
        goal_pos = [box_pos[0]+self.d*rand_dir[0], box_pos[1]+self.d*rand_dir[1]]

        self.goal.x = goal_pos[0]
        self.goal.y = goal_pos[1]

        # Generates the agents position TODO: avoid collisions
        for i in range(self.n_agents):
            rand_rad = random.uniform(-2*np.pi, 2*np.pi)
            rand_dir = [np.cos(rand_rad), np.sin(rand_rad)]
            self.agents[i].agent_rigid_body.position = (
                box_pos[0]+self.goal_radius*rand_dir[0], box_pos[1]+self.goal_radius*rand_dir[1])


    def getLastObjPosition(self):
        return self.obj.last_obj_pos

    def getLastAgentPosition(self, agent_id):
        return self.agents[agent_id].last_agent_pos

    def getObjPosition(self):
        return self.obj.obj_rigid_body.position

    def getAgentPosition(self, agent_id):
        return self.agents[agent_id].agent_rigid_body.position

    def distToObjective(self):
        return (self.goal - self.obj.obj_rigid_body.position).length

    def distToObject(self, agent_id):
        return (self.agents[agent_id].agent_rigid_body.position - self.obj.obj_rigid_body.position).length

    def update(self, timeStep, velocity_iterator, position_iterator):
        for i in range(self.n_agents):
            self.agents[i].Update()
        self.obj.Update()
        self.world.Step(timeStep=timeStep, velocityIterations=velocity_iterator, positionIterations=position_iterator)
        self.world.ClearForces()

    def worldToScreen(self, position):
        return (int(position[0] * self.pixels_per_meter), int(position[1] * self.pixels_per_meter))

    def getStateImg(self, agent_id):
        agent = self.agents[agent_id]

        # clear previous buffer
        screen = np.zeros(shape=self.observed_dist_shape, dtype=np.float32)

        # screen coordinates
        agent_center = b2Vec2(self.observed_dist_shape[0]/2.0,self.observed_dist_shape[1]/2.0)
        agent_center_point = (int(agent_center.x),int(agent_center.y))

        # world coordinates in the visibility window
        object_to_agent = self.obj.obj_rigid_body.position - agent.agent_rigid_body.position + agent_center / self.pixels_per_meter
        objective_to_agent = self.goal - agent.agent_rigid_body.position# + agent_center / self.pixels_per_meter
        objective_to_agent.Normalize()
        objective_to_agent = agent.agent_rigid_body.position + objective_to_agent * 1000.0

        # screen coordinates
        obj_g_screen = self.worldToScreen((objective_to_agent.x, objective_to_agent.y))
        #screen_pos = self.worldToScreen((objective_to_agent.x, objective_to_agent.y))
        #cv2.circle(screen, screen_pos, int(self.goal_radius*self.pixels_per_meter), (0.0, 1.0, 0.0), thickness=5, lineType=4)

        # draw all shapes and objects
        screen_pos = self.worldToScreen((object_to_agent.x, object_to_agent.y))
        cv2.circle(screen, screen_pos, int(self.obj_proximity_radius*self.pixels_per_meter), (0, 1, 0, 0), thickness=5, lineType=4)
        cv2.circle(screen, screen_pos, int(self.obj.obj_radius*self.pixels_per_meter), (0, 0, 1, 0), -1)
        # cv2.circle(screen, agent_center_point, int(agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)
        cv2.line(screen, agent_center_point, obj_g_screen, color=(1,0,0,0), thickness=5)
        output_img = np.zeros(shape=self.state_shape, dtype=np.float32)
        output_img = cv2.resize(screen, dsize=(self.state_shape[0], self.state_shape[1]), interpolation = cv2.INTER_AREA)
        
        '''
        Enhance each channel and clip between 0 and 1 to 
        avoid noise in the network inputs
        '''
        # output_img[:,:,0] = np.clip(output_img[:,:,0] * 1000.0, 0, 1.0)
        #output_img[:,:,1] = np.clip(output_img[:,:,1] * 1000.0, 0, 1.0)
        #output_img[:,:,2] = np.clip(output_img[:,:,2] * 1000.0, 0, 1.0)

        '''
        Make objective direction more visible over obstacle
        '''
        # output_img[:,:,2] = output_img[:,:,2] - output_img[:,:,0]
        output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        output_gray += output_img[:,:,0] * 10
        output_gray = np.clip(output_gray * 2.0, 0.0, 1.0)
        #output_gray = output_gray - output_gray * 0.5
        # output_gray[:,:] = output_gray[:,:] + output_img[:,:,0]
        output_gray[7,7] = 1.0
        output_gray[7,8] = 1.0
        output_gray[8,7] = 1.0
        output_gray[8,8] = 1.0

        return output_gray

    def drawToBuffer(self):
        # clear previous buffer
        self.screen = np.ones(shape=(self.height, self.width, 3), dtype=np.float32)

        # draw all shapes and objects
        screen_pos = self.worldToScreen((self.goal.x, self.goal.y))
        cv2.circle(self.screen, screen_pos, int(self.goal_radius*self.pixels_per_meter), (0.0, 1.0, 0.0), thickness=4, lineType=4)

        screen_pos = self.worldToScreen(self.obj.GetPositionAsList())
        cv2.circle(self.screen, screen_pos, int(self.obj_proximity_radius*self.pixels_per_meter), (0, 0.5, 0.5, 0), thickness=4, lineType=4)

        screen_pos = self.worldToScreen(self.obj.GetPositionAsList())
        cv2.circle(self.screen, screen_pos, int(self.obj.obj_radius*self.pixels_per_meter), (0, 0, 1, 0), -1)

        for i in range(self.n_agents):
            agent = self.agents[i]
            screen_pos = self.worldToScreen(agent.GetPositionAsList())
            cv2.circle(self.screen, screen_pos, int(agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)

        return self.screen