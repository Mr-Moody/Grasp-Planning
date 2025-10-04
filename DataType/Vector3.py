import math

class Vector3():
    def __init__(self, x:float=0, y:float=0, z:float=0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        if not isinstance(other, Vector3):
            raise ValueError("Can only add Vector3 with other Vector3")
        
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        if not isinstance(other, Vector3):
            raise ValueError("Can only subtract Vector3 with other Vector3")
        
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        
        raise ValueError("Can only scalar multiply by float and int, or elementwise multiplication by another Vector3.")
    
    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Vector3(self.x / other, self.y / other, self.z / other)
        
        if isinstance(other, Vector3):
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        
        raise ValueError("Can only scalar multiply by float and int, or elementwise multiplication by another Vector3.")
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector3):
            raise ValueError("Can only compare equality between two Vector3 objects.")
        
        if (self.x == other.x) and (self.y == other.y) and (self.z == other.z):
            return True
            
        return False
    
    def __iter__(self):
        """
        Needed as pybullet expects a list for position and velocity
        """
        return iter((self.x, self.y, self.z))
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalise(self):
        magnitude = self.magnitude()

        if magnitude == 0:
            return Vector3(0,0,0)
        
        return self / magnitude
    
    @staticmethod
    def dot(v1, v2) -> float:
        if not isinstance(v1, Vector3) or not isinstance(v2, Vector3):
            raise ValueError("Can only take dot product of two Vector3 objects.")
        
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    
    @staticmethod
    def cross(v1, v2):
        if not isinstance(v1, Vector3) or not isinstance(v2, Vector3):
            raise ValueError("Can only take cross product of two Vector3 objects.")
        
        return Vector3(v1.y * v2.z - v1.z * v2.y,
                       v1.z * v2.x - v1.x * v2.z,
                       v1.x * v2.y - v1.y * v2.x)
    

if __name__ == "__main__":
    a = Vector3(1,2,3)
    b = Vector3(4,5,6)

    print(Vector3.cross(a,b))
    print(Vector3.dot(a,b))