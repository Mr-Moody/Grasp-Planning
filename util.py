import pybullet as p
import numpy as np

def drawGizmo(position:np.ndarray=np.array([0,0,0]), scale=0.005, color=[0, 0, 0, 1]):
    sphere = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=scale,
        rgbaColor=color  # black by default
    )
    
    p.createMultiBody(baseVisualShapeIndex=sphere, basePosition=position)