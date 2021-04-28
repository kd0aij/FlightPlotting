import numpy as np
import plotly.graph_objects as go
from geometry import Transformation, Coord, Quaternion, Point, Quaternions, Points

class OBJ(object):
    def __init__(self, vertices: Points, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces
    
    @staticmethod
    def from_obj_data(odata):
        """[summary]
        lifted in its entirity from here: https://chart-studio.plotly.com/~empet/15040/plotly-mesh3d-from-a-wavefront-obj-f/#/
        odata is the string read from an obj file
        """
        vertices = []
        faces = []
        lines = odata.splitlines()

        for line in lines:
            slist = line.split()
            if slist:
                if slist[0] == 'v':
                    vertex = np.array(slist[1:], dtype=float)
                    vertices.append(vertex)
                elif slist[0] == 'f':
                    face = []
                    for k in range(1, len(slist)):
                        face.append([int(s)
                                    for s in slist[k].replace('//', '/').split('/')])
                    if len(face) > 3:  # triangulate the n-polyonal face, n>3
                        faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1]
                                    for k in range(1, len(face)-1)])
                    else:
                        faces.append([face[j][0]-1 for j in range(len(face))])
                else:
                    pass

        return OBJ(Points(np.array(vertices)), np.array(faces))

    @staticmethod
    def from_obj_file(file):
        if isinstance(file, str):
            file = open(file, encoding="utf-8")
        obj_data = file.read()        
        return OBJ.from_obj_data(obj_data)

    def transform(self, transformantion: Transformation=Transformation(Point(0.75, 0, 0), Quaternion.from_euler(Point(np.pi, 0, -np.pi/2)))):
        return OBJ(transformantion.point(self.vertices), self.faces)
    
    def scale(self, scale_factor):
        return OBJ(self.vertices * scale_factor, self.faces)

    def create_mesh(self, colour='orange', name: str = ''):
        """Generate a Mesh3d of my plane transformed by the requested transformation.

        Args:
            name (str, optional): The name of the series. Defaults to ''.

        Returns:
            go.Mesh3d: a plotly Mesh3d containing the model
        """

        x, y, z = self.vertices.data[:, :3].T
        I, J, K = self.faces.T
        return go.Mesh3d(
            x=x, y=y, z=z, i=I, j=J, k=K,
            name=name,
            showscale=False,
            hoverinfo="name",
            color=colour
        )  # vertexcolor=vertices[:, 3:], #the color codes must be triplets of floats  in [0,1]!!
