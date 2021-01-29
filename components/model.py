import numpy as np
import plotly.graph_objects as go
from geometry import Transformation, Coord, Quaternion, Point, Quaternions, Points


with open('data/model.obj', encoding="utf-8") as f:
    obj_data = f.read()


def obj_data_to_mesh3d(odata):
    # odata is the string read from an obj file
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
                    face.append([int(s) for s in slist[k].replace('//','/').split('/')])
                if len(face) > 3: # triangulate the n-polyonal face, n>3
                    faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
                else:    
                    faces.append([face[j][0]-1 for j in range(len(face))])
            else: pass
    
    
    return np.array(vertices), np.array(faces)  

vertices, faces = obj_data_to_mesh3d(obj_data)

pins=Points(vertices)
pouts = Quaternions.from_quaternion(Quaternion.from_euler(Point(np.pi,0,-np.pi/2)), pins.count).transform_point(pins)
pouts = Point(0.75,0,0) + pouts 

def create_mesh(transform, scale=10):
    plotpoints=transform.point(scale * pouts)

    x, y, z = plotpoints.data[:,:3].T
    I, J, K = faces.T
    return go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,name='',showscale=False) #vertexcolor=vertices[:, 3:], #the color codes must be triplets of floats  in [0,1]!!                      

