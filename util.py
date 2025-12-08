import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from typing import Optional
from constants import TIME, TICK_RATE, NUM_TICKS, GUI

global_gui = False

def setupEnvironment(gui:Optional[bool]=None):
    global global_gui

    # Connect to physics server
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        if gui is None:
            gui = GUI

        global_gui = gui

        if global_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load plane.urdf before adding Models path to avoid search path conflicts
    plane_urdf_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
    plane_id = p.loadURDF(plane_urdf_path)
    
    # Add Models path - use absolute path to avoid issues with spaces in path
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Models"))
    if os.path.exists(model_path) and os.path.isdir(model_path):
        p.setAdditionalSearchPath(model_path)
    
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81)
    
    # # Set physics solver parameters for better contact stability
    # p.setPhysicsEngineParameter(
    #     numSolverIterations=150,  # More iterations for better contact resolution
    #     contactBreakingThreshold=0.00001,  # Much lower threshold for better contact maintenance
    #     enableConeFriction=1,  # Enable cone friction for more realistic contacts
    #     restitutionVelocityThreshold=0.01  # Lower threshold to reduce bouncing
    # )

    return plane_id

def pause(wait_time:float):
    """
    Run simulation for time seconds.
    """
    num_steps = int(wait_time / TICK_RATE)

    for _ in range(num_steps):
        p.stepSimulation()
        
        if global_gui:
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