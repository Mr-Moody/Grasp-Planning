import pybullet as p

from DataType.Vector3 import Vector3

def drawGizmo(position:Vector3=Vector3.zero(), scale=0.005, color=[0, 0, 0, 1]):
    if position:
        sphere = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=scale,
            rgbaColor=color  # black by default
        )
        
        p.createMultiBody(baseVisualShapeIndex=sphere, basePosition=list(position))