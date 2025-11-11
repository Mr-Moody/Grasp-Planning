import pybullet as p
import pybullet_data
import numpy as np
import time
import os

from constants import TIME, TICK_RATE, NUM_TICKS

def setupEnvironment():
    # Connect to physics server
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        p.connect(p.GUI)

    
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81)
    
    # # Set physics solver parameters for better contact stability
    # p.setPhysicsEngineParameter(
    #     numSolverIterations=150,  # More iterations for better contact resolution
    #     contactBreakingThreshold=0.00001,  # Much lower threshold for better contact maintenance
    #     enableConeFriction=1,  # Enable cone friction for more realistic contacts
    #     restitutionVelocityThreshold=0.01  # Lower threshold to reduce bouncing
    # )

    # Create floor plane
    plane_id = p.loadURDF("plane.urdf")

    return plane_id

def pause(wait_time:float):
    """
    Run simulation for time seconds.
    """
    t0 = time.time()
    while ((time.time() - t0) < wait_time):
        p.stepSimulation()
        time.sleep(TICK_RATE)

def drawGizmo(position:np.ndarray=np.array([0,0,0]), scale:float=0.005, color:list[float]=[0, 0, 0, 1]) -> int:
    sphere = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=scale,
        rgbaColor=color  # black by default
    )
    
    gizmo_id = p.createMultiBody(baseVisualShapeIndex=sphere, basePosition=position)

    return gizmo_id

def removeGizmo(gizmo_id:int) -> None:
    if gizmo_id is not None:
        p.removeBody(gizmo_id)