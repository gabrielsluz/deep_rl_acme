from random import randint
import random
import cv2
import numpy as np
from Box2D import b2World, b2Vec2, b2Transform
from research_envs.b2PushWorld.Agent import Agent
from research_envs.b2PushWorld.Object import CircleObj, RectangleObj, PolygonalObj

class PushSimulator:
    def __init__(self, pixelsPerMeter = 20, width = 1024, height = 1024, 
                 objectiveRadius = 3.0, objProxRadius=15, objList=None):
        # opencv is used for simple rendering to avoid
        # box2D framework
        self.pixels_per_meter = pixelsPerMeter
        self.width = width
        self.height = height
        self.screen_width = int(width*2)
        self.screen_height = int(height*2)
        self.screen = np.zeros(shape=(self.screen_height, self.screen_width), dtype=np.float32)
        self.state_shape = (16,16,1)
        self.observed_dist_shape = (320,320,3)

        # ----------- world creation ----------
        # crate a simple box2D world with -9.8 gravity acceleration
        self.world = b2World(gravity=(0, 0.0), doSleep=False)

        # ----------- specify goal position -----------
        self.goal_radius = objectiveRadius
        self.obj_proximity_radius = objProxRadius
        self.goal = b2Vec2(0,0)
        self.goal_orientation = 0.0

        # ----------- specify objects -----------
        if objList is None:
            self.obj_l = [
                CircleObj(simulator=self, x=25, y=25, radius=4.0),
                RectangleObj(simulator=self, x=25, y=25, height=10, width=5),
                PolygonalObj(simulator=self, x=25, y=25, vertices=[(5,10), (0,0), (10,0)])
            ]
        else:
            self.obj_l = objList
        self.obj = self.obj_l[random.randrange(0, len(self.obj_l))]
        self.agent = Agent(simulator=self, x=30, y=25, radius=1.0, velocity=2.0, forceLength=2.0, totalDirections=8)

        # Define object - goal - robot reset distances
        self.max_dist_obj_goal = 30
        self.min_dist_obj_goal = 2
        self.max_ori_obj_goal = np.pi / 2

    def reset(self):
        # Limits the box distance to goal and orientation difference.
        # Create object
        self.obj = self.obj_l[random.randrange(0, len(self.obj_l))]

        max_x = int(self.width/self.pixels_per_meter)
        max_y = int(self.height/self.pixels_per_meter)
        # box_pos = [random.uniform(0, max_x), random.uniform(0, max_y)]
        box_pos = [
            random.uniform(self.max_dist_obj_goal, max_x + self.max_dist_obj_goal), 
            random.uniform(self.max_dist_obj_goal, max_y + self.max_dist_obj_goal)]

        self.obj.obj_rigid_body.position = (box_pos[0], box_pos[1])
        self.obj.obj_rigid_body.angle = random.uniform(0, 2*np.pi)

        # Create goal based on object
        rand_rad = random.uniform(-2*np.pi, 2*np.pi)
        rand_dist = random.uniform(self.min_dist_obj_goal, self.max_dist_obj_goal)
        self.goal.x = box_pos[0] + rand_dist * np.cos(rand_rad)
        self.goal.y = box_pos[1] + rand_dist * np.sin(rand_rad)

        self.goal_orientation = self.obj.obj_rigid_body.angle + random.uniform(
            -self.max_ori_obj_goal, self.max_ori_obj_goal)

        # Obter posicao do robo
        rand_rad = random.uniform(-2*np.pi, 2*np.pi)
        rand_dir = [np.cos(rand_rad), np.sin(rand_rad)]
        self.agent.agent_rigid_body.position = (
            box_pos[0] + 5*rand_dir[0], box_pos[1] + 5*rand_dir[1])


    def getLastObjPosition(self):
        return self.obj.last_obj_pos

    def getLastAgentPosition(self):
        return self.agent.last_agent_pos

    def getObjPosition(self):
        return self.obj.obj_rigid_body.position

    def getAgentPosition(self):
        return self.agent.agent_rigid_body.position

    def distToObjective(self):
        return (self.goal - self.obj.obj_rigid_body.position).length

    def distToObject(self):
        return (self.agent.agent_rigid_body.position - self.obj.obj_rigid_body.position).length

    def distToOrientation(self):
        if self.obj.obj_type == 'Circle': # Does not work well with circles
            return 0.0
        else:
            # Calculate the angle between the object and the goal
            obj_angle = self.obj.obj_rigid_body.angle % (2*np.pi)
            angle_diff = self.goal_orientation - obj_angle
            if angle_diff > np.pi:
                angle_diff -= 2*np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2*np.pi
            return angle_diff
            

    def update(self, timeStep, velocity_iterator, position_iterator):
        self.agent.Update()
        self.obj.Update()
        self.world.Step(timeStep=timeStep, velocityIterations=velocity_iterator, positionIterations=position_iterator)
        self.world.ClearForces()

    def worldToScreen(self, position):
        return (int(position[0] * self.pixels_per_meter), int(position[1] * self.pixels_per_meter))

    def getStateImg(self):
        # clear previous buffer
        screen = np.zeros(shape=self.observed_dist_shape, dtype=np.float32)

        # screen coordinates
        agent_center = b2Vec2(self.observed_dist_shape[0]/2.0,self.observed_dist_shape[1]/2.0)
        agent_center_point = (int(agent_center.x),int(agent_center.y))

        # world coordinates in the visibility window
        object_to_agent = self.obj.obj_rigid_body.position - self.agent.agent_rigid_body.position + agent_center / self.pixels_per_meter
        objective_to_agent = self.goal - self.agent.agent_rigid_body.position# + agent_center / self.pixels_per_meter
        objective_to_agent.Normalize()
        objective_to_agent = self.agent.agent_rigid_body.position + objective_to_agent * 1000.0

        # screen coordinates
        obj_g_screen = self.worldToScreen((objective_to_agent.x, objective_to_agent.y))
        #screen_pos = self.worldToScreen((objective_to_agent.x, objective_to_agent.y))
        #cv2.circle(screen, screen_pos, int(self.goal_radius*self.pixels_per_meter), (0.0, 1.0, 0.0), thickness=5, lineType=4)

        # draw all shapes and objects
        screen_pos = self.worldToScreen((object_to_agent.x, object_to_agent.y))
        cv2.circle(screen, screen_pos, int(self.obj_proximity_radius*self.pixels_per_meter), (0, 1, 0, 0), thickness=5, lineType=4)
        self.obj.DrawInPos(screen_pos, self.pixels_per_meter, screen, (0, 0, 1, 0), -1)
        # cv2.circle(screen, screen_pos, int(self.obj.obj_radius*self.pixels_per_meter), (0, 0, 1, 0), -1)
        # cv2.circle(screen, agent_center_point, int(self.agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)
        cv2.line(screen, agent_center_point, obj_g_screen, color=(1,0,0,0), thickness=5)
        output_img = np.zeros(shape=self.state_shape, dtype=np.float32)
        output_img = cv2.resize(screen, dsize=(self.state_shape[0], self.state_shape[1]), interpolation = cv2.INTER_AREA)

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

    def drawArrow(self, image, world_pos, angle, len, color):
        start_pos = self.worldToScreen(world_pos)
        end_pos = (world_pos[0] + len * np.cos(angle), world_pos[1] + len * np.sin(angle))
        end_pos = self.worldToScreen(end_pos)
        cv2.arrowedLine(image, start_pos, end_pos, color, thickness=6)

    def drawObjInGoal(self, image, color=(0.0, 1.0, 0.0)):
        # Draw the object in the goal location with the goal orientation
        if self.obj.obj_type == 'Circle':
            screen_pos = self.worldToScreen((self.goal.x, self.goal.y))
            cv2.circle(
                self.screen, screen_pos, int(self.obj.obj_radius*self.pixels_per_meter),color,
                thickness=4, lineType=4)
            return
        # Rotate and Translate
        transform_matrix = b2Transform()
        transform_matrix.SetIdentity()
        transform_matrix.Set(self.goal, self.goal_orientation)
        vertices = [transform_matrix * v for v in self.obj.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)
        self.drawArrow(image, self.goal, self.goal_orientation, 10, color)
        # cv.drawContours(	image, contours, contourIdx, color[, thickness)

    def drawToBuffer(self):
        # clear previous buffer
        self.screen = np.ones(shape=(self.screen_height, self.screen_width, 3), dtype=np.float32)

        # draw all shapes and objects
        # Draw the object in the goal location with the goal orientation
        self.drawObjInGoal(self.screen)

        screen_pos = self.worldToScreen(self.obj.GetPositionAsList())
        cv2.circle(self.screen, screen_pos, int(self.obj_proximity_radius*self.pixels_per_meter), (0, 0.5, 0.5, 0), thickness=4, lineType=4)

        self.obj.Draw(self.pixels_per_meter, self.screen, (0, 0, 1, 0), -1)
        self.drawArrow(self.screen, self.obj.GetPositionAsList() , self.obj.obj_rigid_body.angle, 10, (0, 0, 1, 0))

        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        cv2.circle(self.screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)

        return self.screen


    # Returns the action that is closest to the direction agent to object
    def getClosestActionToObject(self):
        # Get the direction from the agent to the object
        direction = self.getObjPosition() - self.getAgentPosition()
        direction.Normalize()
        # Get action directions
        actions_directions = [self.agent.ActionIdToForce(i) for i in range(self.agent.directions)]
        # Compute inner product between direction and action directions
        inner_products = [np.dot(direction, action_direction) for action_direction in actions_directions]
        action = np.argmax(inner_products)
        return action