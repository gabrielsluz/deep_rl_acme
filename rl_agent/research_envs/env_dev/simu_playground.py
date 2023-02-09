"""
A playground for testing the PushSimulator
"""
# To execute from rl_agent
import sys
sys.path.append('.')

from research_envs.b2PushWorld.Object import CircleObj, RectangleObj, PolygonalObj
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer
from research_envs.b2PushWorld.Agent import Agent

from Box2D import b2World
import numpy as np
import cv2

# Criar um loop que renderiza na tela e roda o simulador => controlar a bolinha do robô com as setsas

class Simulator:
    def __init__(self):
        self.world = b2World(gravity=(0, 0.0), doSleep=False)
        self.timestep = 1.0 / 60.0
        self.vel_iterator = 6
        self.pos_iterator = 2

        # self.obj = CircleObj(self, 10, 10, 4)
        # self.obj = RectangleObj(self, 10, 10, 10, 4)
        self.obj = PolygonalObj(simulator=self, x=25, y=25, vertices=[(6,10), (0,0), (10,0)])
        # self.obj = PolygonalObj(self, 10, 10, [(0,2), (3,0), (5,2), (4,4), (1,4)])
        print(self.obj.obj_rigid_body.transform.q)

        self.agent = Agent(simulator=self, x=20, y=8, radius=1.0, velocity=10.0, forceLength=1.0, totalDirections=4)

        self.height = 1024
        self.width = 1024
        self.pixels_per_meter = 20
        self.screen = np.zeros(shape=(self.height, self.width), dtype=np.float32)

    def update(self):
        self.agent.Update()
        self.obj.Update()
        self.world.Step(timeStep=self.timestep, velocityIterations=self.vel_iterator, positionIterations=self.pos_iterator)
        self.world.ClearForces()

    def worldToScreen(self, position):
        return (int(position[0] * self.pixels_per_meter), int(position[1] * self.pixels_per_meter))

    def drawToBuffer(self):
        # clear previous buffer
        self.screen = np.ones(shape=(self.height, self.width, 3), dtype=np.float32)

        self.obj.Draw(self.pixels_per_meter, self.screen, (0, 0, 1, 0), -1)

        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        cv2.circle(self.screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)

        agent_pos = self.worldToScreen(self.agent.GetPositionAsList())
        obj_pos = self.worldToScreen(self.obj.GetPositionAsList())
        cv2.line(self.screen, agent_pos, obj_pos, (0, 255, 0), thickness=2)

        return self.screen

    def Keyboard(self, key):
        if key == 97: # a
            self.agent.PerformAction(2)
        elif key == 115: # s
            self.agent.PerformAction(1)
        elif key == 100: # d
            self.agent.PerformAction(0)
        elif key  == 119: # w
            self.agent.PerformAction(3)

if __name__ == "__main__":
    simulator = Simulator()
    scene_buffer = CvDrawBuffer(window_name="Push Simulation", resolution=(1024,1024))

    for i in range(1000000):
        # Simu
        dt = 1.0 / 60.0 #1.0 / 60.0
        key = 0xFF & cv2.waitKey(int(dt * 1000.0))
        if key == 27:
            break
        simulator.Keyboard(key)
        if key != 255: print(key)
        # simulator.agent.PerformAction(4)
        simulator.update()
        async_buffer = simulator.drawToBuffer()
        scene_buffer.PushFrame(async_buffer)
        scene_buffer.Draw()
        # print('Ang:', simulator.obj.obj_rigid_body.angle)
        # print('OBJ:', simulator.obj.GetPositionAsList(), " AGNT:", simulator.agent.GetPositionAsList())
        # print('DIST:', (simulator.agent.agent_rigid_body.position - simulator.obj.obj_rigid_body.position).length)
        # cv2.waitKey(1)