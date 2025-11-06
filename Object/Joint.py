import pybullet as p

class Joint():
    MAX_VELOCITY = 2
    MAX_FORCE = 200

    def __init__(self, body_id, index:int, position:float=0, min_limit:float=float("-INF"), max_limit:float=float("INF")):
        self.body_id = body_id
        self.index = index
        self.position = position # position is angle in radians
        self.min_limit = min_limit
        self.max_limit = max_limit

        p.resetJointState(self.body_id, self.index, self.position)

    def move(self, position) -> None:
        """
        Move joint within limits of joint.
        """
        if self.min_limit is not None:
            position = max(position, self.min_limit)

        if self.max_limit is not None:
            position = min(position, self.max_limit)

        self.position = position

        p.setJointMotorControl2(self.body_id, 
                                self.index, 
                                p.POSITION_CONTROL,
                                targetPosition=self.position, 
                                maxVelocity=self.MAX_VELOCITY, 
                                force=self.MAX_FORCE)

    def getPosition(self) -> float:
        """
        Get joint position from pybullet.
        """
        state = p.getJointState(self.body_id, self.index)
        self.position = state[0]

        return self.position
