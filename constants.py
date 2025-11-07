import math
import os

TIME = 10
TICK_RATE = 1./240.
NUM_TICKS = math.ceil(TIME / TICK_RATE)
PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")