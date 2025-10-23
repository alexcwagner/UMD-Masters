import struct
import numpy as np
import sys

VERT_DTYPE = np.dtype([
    ("pos", np.float32, 3),
    ("nrm", np.float32, 3),
    ("col", np.float32, 3),
])

class Scene:
    def __init__(self):
        self.meshes = []
        self.lights = []
    def add_mesh(self, mesh):
        self.meshes.append(mesh)
  
class Vertex:
    def __init__(self, mesh, coord, normal=None):
        self.mesh = mesh
        self.index = None
        self.coord = np.array(coord, dtype=np.float32)
        self.normal = None if normal is None else np.array(normal, dtype=np.float32)
        self.color = None
        self.face_normals = []
        
    def __hash__(self):
        
        hashable = (tuple(self.coord), 
                    None if self.normal is None else tuple(self.normal))
        return hash(hashable)        

    def __eq__(self, other):
        try:
            if not isinstance(other, Vertex):
                return False
            
            return (np.array_equal(self.coord, other.coord) 
                    and np.array_equal(self.normal, other.normal))
        except Exception as e:
            print(self.coord, other.coord, self.normal, other.normal)
            raise e

    def __repr__(self):
        return f"Vertex({self.coord}, {self.normal})"    
  
    
class Mesh:
    def __init__(self):
        #self.vertex_manager = VertexManager(self)
        #self.faces = Faces(self)
        
        self.vertex_list = []
        self.vertex_lookup = {}
        self.face_list = []
        
        
        pass
    
    def register_vertex(self, coord, normal):
        vertex = Vertex(self, coord, normal)
        #print(f"vertex: {vertex}")
        if vertex in self.vertex_lookup:
            #print(f"found! returning {self.vertex_lookup[vertex]}")
            return self.vertex_lookup[vertex]
        index = len(self.vertex_list)
        vertex.index = index
        self.vertex_list.append(vertex)
        self.vertex_lookup[vertex] = index
        #print(f"new vertex! returning {index}")
        return index
    
    def add_face(self, coords, flat=True):
        
        normal = self.face_normal(coords)
            
        if flat:
            face = [ self.register_vertex(coord, normal) for coord in coords ]
        else:
            face = [ self.register_vertex(coord, None) for coord in coords ]
        
        for idx in face:
            self.vertex_list[idx].face_normals.append(normal)
        
        self.face_list.append(face)
        return face
        
    
    @classmethod
    def face_normal(cls, coords):
        v = np.asarray(coords)
        n = np.zeros(3)
        for i in range(len(v)):
            v_curr = v[i]
            v_next = v[(i + 1) % len(v)]
            n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
            n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
            n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])
        return n / np.linalg.norm(n)
        

    def get_buffer_data(self):
    
        colors = None
        
        positions = np.array([v.coord for v in self.vertex_list], dtype=np.float32)
        
        colors = np.array([v.color or (0.8, 0.8, 0.8) for v in self.vertex_list],
                          dtype=np.float32)
        
        normals = [] # np.array([v.normal for v in self.vertex_list], dtype=np.float32)
        for idx, vtx in enumerate(self.vertex_list):
            #print(idx)
            if vtx.normal is None:
                #for normal in vtx.face_normals:
                #    print(normal)
                normal_sum = sum(vtx.face_normals)
                avg_normal = normal_sum/np.linalg.norm(normal_sum)
                #print(avg_normal)
                #print()
                normals.append(avg_normal)
                
            else:
                normals.append(vtx.normal)
        normals = np.array(normals)
        
        indices = []
        
        for face in self.face_list:
            for idx in range(1, len(face)-1):
                indices.extend( [face[0], face[idx], face[idx+1] ] )
        indices = np.array(indices, dtype=np.uint32)
        
        
        # Interleave the data
        
        data = np.empty(len(positions), dtype=VERT_DTYPE)
        data['pos'] = np.asarray(positions, dtype=np.float32)
        data['nrm'] = np.asarray(normals, dtype=np.float32)
        data['col'] = np.asarray(colors, dtype=np.float32)
        
        
        #return positions, normals, indices, colors
        return data, indices


    @classmethod
    def test_cube(cls, flat=True):
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
            mesh.add_face(face, flat=flat)
            

        return mesh
    
    @classmethod
    def from_STL(cls, path, flat=True):
        mesh = cls()
        with open(path, "rb") as f:
            f.read(80)  # header
            tri_count = struct.unpack("<I", f.read(4))[0]

            for _ in range(tri_count):
                _ = struct.unpack("<fff", f.read(12))
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

                #face_indices = mesh.add_face(coords=(v1, v2, v3), flat=True)
                v1 = [x/200 for x in v1]
                v2 = [x/200 for x in v2]
                v3 = [x/200 for x in v3]
                face_indices = mesh.add_face(coords=(v1, v2, v3), flat=flat)
                
                for idx in face_indices:
                    vtx = mesh.vertex_list[idx]
                    vtx.color = color

        return mesh
    
print("Module 'Mesh' Loaded")  

if __name__ == '__main__':
    
    #mesh = Mesh.test_cube()
    mesh = Mesh.from_STL('../resources/72-gon.stl', flat=False)
    print(mesh.get_buffer_data())
    
