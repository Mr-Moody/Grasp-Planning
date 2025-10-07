from DataType.Vector3 import Vector3

class CubeSphere():
    def __init__(self,origin:Vector3=Vector3.zero(),radius:float=0, resolution:int=5) -> None:
        self.origin = origin
        self.radius = radius
        self.resolution = resolution

        #reference direction of centre of face from origin of sphere top: z, left: -x, right: x, front: -y, back: y
        self.top = CubeFace(self.origin, Vector3(0,0,self.radius), self.radius, self.resolution, Vector3(0,1,0), Vector3(1,0,0))
        self.front = CubeFace(self.origin, Vector3(0,-self.radius,0), self.radius, self.resolution, Vector3(0,0,1), Vector3(1,0,0))
        self.back = CubeFace(self.origin, Vector3(0,self.radius,0), self.radius, self.resolution, Vector3(0,0,1), Vector3(-1,0,0))
        self.left = CubeFace(self.origin, Vector3(-self.radius,0,0), self.radius, self.resolution, Vector3(0,0,1), Vector3(-1,0,0))
        self.right = CubeFace(self.origin, Vector3(self.radius,0,0), self.radius, self.resolution, Vector3(0,0,1), Vector3(0,-1,0))

class CubeFace():
    def __init__(self,
                 origin:Vector3=Vector3.zero(),
                 offset:Vector3=Vector3.zero(), 
                 radius:float=0, 
                 resolution:int=0, 
                 local_up:Vector3=Vector3.zero(), 
                 local_right:Vector3=Vector3.zero()) -> None:
        
        self.origin = origin
        self.offset = offset
        self.radius = radius
        self.resolution = resolution
        self.local_up = local_up
        self.local_right = local_right

        self.vertices = self.generateVertices()

    def generateVertices(self):
        spacing = self.radius / (self.resolution - 1)

        vertices: list[list[Vector3]] = [[Vector3() for _ in range(self.resolution)] for _ in range(self.resolution)]

        print(vertices)

        for i in range(self.resolution -1):
            for j in range(self.resolution -1):
                vertices[i][j] = self.origin + self.offset + i*spacing*self.local_right + j*spacing*self.local_up


if __name__ == "__main__":
    c = CubeSphere(Vector3.zero(), 1, 5)

    print(c.top.vertices)