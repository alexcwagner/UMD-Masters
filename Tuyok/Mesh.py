import struct
import numpy as np

class Scene:
    def __init__(self):
        self.meshes = []
        self.lights = []
    def add_mesh(self, mesh):
        self.meshes.append(mesh)
        
class Mesh:
    def __init__(self):
        self.vertices = Vertices(self)
        self.faces = Faces(self)

    def get_buffer_data(self):
        
        
        colors = None
        
        vertices = np.array([v.point for v in self.vertices.vertices], dtype=np.float32)
        normals = np.array([v.normal for v in self.vertices.vertices], dtype=np.float32)
        colors = np.array([v.color or (0.8, 0.8, 0.8) for v in self.vertices.vertices],
                          dtype=np.float32)
        indices = []
        
        for face in self.faces.faces:
            for idx in range(1, len(face.indices)-1):
                indices.extend( [face.indices[0], face.indices[idx], face.indices[idx+1] ] )
        indices = np.array(indices, dtype=np.uint32)
        return vertices, normals, indices, colors


    @classmethod
    def test_cube(cls):
        mesh = Mesh()
        
        faces = (
                ((-1, -1, -1), (-1, +1, -1), (+1, +1, -1), (+1, -1, -1)),
                ((-1, -1, -1), (-1, -1, +1), (-1, +1, +1), (-1, +1, -1)),
                ((-1, -1, -1), (+1, -1, -1), (+1, -1, +1), (-1, -1, +1)),
                ((+1, +1, +1), (+1, -1, +1), (+1, -1, -1), (+1, +1, -1)),
                ((+1, +1, +1), (-1, +1, +1), (-1, -1, +1), (+1, -1, +1)),
                ((+1, +1, +1), (+1, +1, -1), (-1, +1, -1), (-1, +1, +1)),
            )
        
        for face in faces:
            vertices = [Vertex(mesh, point) for point in face]
            face_obj = Face(mesh, vertices)
            mesh.faces.append(face_obj)

        return mesh
    
    @classmethod
    def from_STL(cls, path):
        mesh = cls()
        with open(path, "rb") as f:
            f.read(80)  # header
            tri_count = struct.unpack("<I", f.read(4))[0]

            for _ in range(tri_count):
                n = struct.unpack("<fff", f.read(12))
                v1 = struct.unpack("<fff", f.read(12))
                v2 = struct.unpack("<fff", f.read(12))
                v3 = struct.unpack("<fff", f.read(12))
                attr = struct.unpack("<H", f.read(2))[0]

                if attr & 0x8000:
                    r = ((attr >> 10) & 0x1F) / 31.0
                    g = ((attr >> 5) & 0x1F) / 31.0
                    b = (attr & 0x1F) / 31.0
                    color = (r, g, b)
                else:
                    color = (0.8, 0.8, 0.8)

                mesh.faces.append(
                    Face(mesh, [Vertex(mesh, v, n, color) for v in (v1, v2, v3)]))
        return mesh
    
    
class Face:
    def __init__(self, mesh, vertices, normal=None, color=None, material=None):
        self.mesh = mesh
        self.vertices = vertices
        self.normal = normal or self.face_normal(vertices)
        self.color = color
        self.material = material # for future use
   
        for vertex in self.vertices:
            vertex.normal = None if vertex.normal is None else self.normal

        self.indices = [mesh.vertices.append(vertex) for vertex in vertices]    

        
    def __repr__(self):
        return f"Face({self.indices}, {self.normal}, {self.color}, {self.material})"
    
    @classmethod
    def face_normal(cls, vertices):
        v = np.asarray([v.point for v in vertices])
        n = np.zeros(3)
        for i in range(len(v)):
            v_curr = v[i]
            v_next = v[(i + 1) % len(v)]
            n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
            n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
            n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])
        return n / np.linalg.norm(n)
    
class Vertex:
    def __init__(self, mesh, point, normal=None, color=None, material=None):
        self.mesh = mesh
        self.point = np.array(point, dtype=np.float32)
        self.normal = None if normal is None else np.array(normal, dtype=np.float32)
        self.color = color
        self.material = material # for future use

    def __hash__(self):
        
        hashable = (tuple(self.point), 
                    self.normal if self.normal is None else tuple(self.normal))
        return hash(hashable)        

    def __eq__(self, other):
        try:
            if not isinstance(other, Vertex):
                return False
            
            return (np.array_equal(self.point, other.point) 
                    and np.array_equal(self.normal, other.normal))
        except Exception as e:
            print(self.point, other.point, self.normal, other.normal)
            raise e

    def __repr__(self):
        return f"Vertex({self.point}, {self.normal})"

class Faces:
    def __init__(self, mesh):
        self.mesh = mesh
        self.faces = []
    def append(self, face):
        self.faces.append(face)


class Vertices:
    def __init__(self, mesh):
        self.mesh = mesh
        self.vertex_lookup = {}
        self.vertices = []
        
    def append(self, vertex, force_creation=False):
        if vertex in self.vertex_lookup:
            return self.vertex_lookup[vertex]
        
        index = len(self.vertices)
        self.vertices.append(vertex)
        self.vertex_lookup[vertex] = index
        return index
    


if __name__ == '__main__':
    print("loading object")
    #mesh = Mesh.from_STL('../resources/72-gon.stl')
    mesh = Mesh.test_cube()

    print(f"vertices: {mesh.vertices.vertices}\n")
    print(f"faces: {mesh.faces.faces}\n")
    print()
    buffer_data = mesh.get_buffer_data()
    for i, buffer in enumerate(buffer_data):
        print(f"{i}: {buffer}\n")
    print("done.")


