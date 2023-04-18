from Box2D import b2CircleShape, b2PolygonShape, b2FixtureDef, b2Vec2, b2ChainShape
import cv2
import math
import numpy as np

class Object:
    # Inheritance: __init__ from derivated class must fill:
    #   obj_type, and obj_shape
    # Must implement the methods: __init__, Draw, DrawInPos
    def __init__(self, simulator = None, x = 0, y = 0):
        # ----------- Object configuration -----------
        self.obj_rigid_body = simulator.world.CreateDynamicBody(
            position=(x,y),
            linearDamping = 2.0,
            angularDamping = 5.0        
        )
        self.obj_fixture_def = b2FixtureDef(shape=self.obj_shape, density=1, friction=0.3)
        self.obj_rigid_body.CreateFixture(self.obj_fixture_def)
        
        # the previous state should be stored inside the environment since
        # it depends on the execution of a full action step until time t+1
        self.last_obj_pos = b2Vec2(x,y)

    def UpdateLastPos(self):
        self.last_obj_pos = self.obj_rigid_body.position.copy()

    def Update(self):
        self.obj_rigid_body.linearVelocity = self.obj_rigid_body.linearVelocity * 0.98

    def GetPositionAsList(self):
        return (self.obj_rigid_body.position[0], self.obj_rigid_body.position[1])

    def worldToScreen(self, position, pixels_per_meter):
        return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))


class CircleObj(Object):
    def __init__(self, simulator = None, x = 0, y = 0, radius=1.0):
        self.obj_type = 'Circle'
        self.obj_radius = radius
        self.obj_shape = b2CircleShape(radius=self.obj_radius)
        super().__init__(simulator, x, y)

    def Draw(self, pixels_per_meter, image, color, thickness):
        position = self.GetPositionAsList()
        screen_pos = self.worldToScreen(position, pixels_per_meter)
        cv2.circle(image, screen_pos, int(self.obj_radius*pixels_per_meter), color, thickness)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        cv2.circle(image, screen_pos, int(self.obj_radius*pixels_per_meter), color, thickness)


class RectangleObj(Object):
    def __init__(self, simulator = None, x = 0, y = 0, height=1.0, width=1.0):
        self.obj_type = 'Rectangle'
        self.height = height
        self.width = width
        self.obj_shape = b2PolygonShape(box=(width/2, height/2))
        super().__init__(simulator, x, y)

    def Draw(self, pixels_per_meter, image, color, thickness):
        vertices = [(self.obj_rigid_body.transform * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        # Only rotate
        vertices = [(self.obj_rigid_body.transform.q * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        # Translate
        vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)


class PolygonalObj(Object):
    def __init__(self, simulator = None, x = 0, y = 0, vertices=[]):
        # Vertices in counterclockwise
        # Adjusts such that centroid is in the center (0,0)
        self.obj_type = 'Polygon'
        aux_obj_shape = b2PolygonShape(vertices=vertices)
        centroid = aux_obj_shape.centroid
        vertices = [(v[0]-centroid[0], v[1]-centroid[1]) for v in vertices]
        self.obj_shape = b2PolygonShape(vertices=vertices)
        super().__init__(simulator, x, y)

    def Draw(self, pixels_per_meter, image, color, thickness):
        vertices = [(self.obj_rigid_body.transform * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        # Only rotate
        vertices = [(self.obj_rigid_body.transform.q * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        # Translate
        vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)


# class ChainShapeObj(Object):
#     def __init__(self, simulator = None, x = 0, y = 0, vertices=[]):
#         # Vertices in counterclockwise
#         # Adjusts such that centroid is in the center (0,0)
#         self.obj_type = 'ChainShape'
#         aux_obj_shape = b2ChainShape(vertices=vertices)
#         centroid = self.compute_centroid(aux_obj_shape)
#         vertices = [(v[0]-centroid[0], v[1]-centroid[1]) for v in vertices]
#         self.obj_shape = b2ChainShape(vertices=vertices)
#         super().__init__(simulator, x, y)
    
#     def compute_centroid(self, obj_shape):
#         # Compute centroid following https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
#         # Assumes vertices are in counterclockwise order
#         # Polygon can be concave
#         # Polygon cannot have holes or self intersection
#         # Polygon must be simple
#         centroid = (0, 0)
#         area = 0
#         for i in range(len(obj_shape.vertices)):
#             v1 = obj_shape.vertices[i]
#             v2 = obj_shape.vertices[(i+1)%len(obj_shape.vertices)]
#             aux = v1[0]*v2[1] - v2[0]*v1[1]
#             area += aux
#             centroid = (centroid[0] + (v1[0] + v2[0])*aux, centroid[1] + (v1[1] + v2[1])*aux)
#         area *= 0.5
#         centroid = (centroid[0]/(6*area), centroid[1]/(6*area))
#         return centroid


#     def Draw(self, pixels_per_meter, image, color, thickness):
#         vertices = [(self.obj_rigid_body.transform * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
#         vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
#         cv2.fillPoly(image, [np.array(vertices)], color)

#     def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
#         # Only rotate
#         vertices = [(self.obj_rigid_body.transform.q * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
#         vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
#         # Translate
#         vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
#         cv2.fillPoly(image, [np.array(vertices)], color)

# Poligono Concavo => feito de partes convexas. Receber lista de lista de vertices. Ou b2ChainShape
# O chain não rotaciona. => talvez setar inertia e mass corretamente.
# Testar com um corpo com joints
# Fazer o draw de cada poligon separadamente
# Como calcular o centroid? => passar por parâmetro. Ou já passar centralizada.