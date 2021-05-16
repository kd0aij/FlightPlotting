import plotly.graph_objects as go
import flightplotting.templates
from .model import OBJ
from geometry import Point, Coord
import numpy as np
from typing import List, Union
from math import cos, sin, tan, radians


# distance from pilot box to runway centerline
# note that origin is on runway centerline
pbox = 15
# max distance from pilot box to aerobatic box
depth = 175
xlim = depth * tan(radians(60))
ylim = depth - pbox
zmin = 20

# maneuvering box: 60 degrees horizontal and vertical from pilot box
def boxtrace():
    return [go.Mesh3d(
        #  0  1     2     3      4    5      6
        x=[0, xlim, 0,    -xlim, xlim, 0,   -xlim], 
        y=[-pbox, ylim, ylim,  ylim, ylim, ylim,  ylim],
        z=[0, 0,    0,     0,    xlim, xlim, xlim], 
        i=[0, 0, 0, 0, 0], 
        j=[1, 2, 1, 3, 4], 
        k=[2, 3, 4, 6, 6],
        opacity=0.4
    )]

# centerline of maneuvering box at depth
def boxplane():
    return [go.Mesh3d(
        #     0      1      2     3
        x=[xlim, -xlim, -xlim, xlim],
        y=[ylim,  ylim,  ylim, ylim],
        z=[zmin,  zmin,  xlim, xlim],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        opacity=0.4
    )]

# maneuvering box over depth range
mindepth = 125
xlim2 = mindepth * tan(radians(60))
ylim2 = mindepth - pbox
def boxfrustum():
    return [go.Mesh3d(
        #     0      1      2     3      4       5       6      7
        x=[xlim, -xlim, -xlim, xlim, xlim2, -xlim2, -xlim2, xlim2],
        y=[ylim,  ylim,  ylim, ylim, ylim2,  ylim2,  ylim2, ylim2],
        z=[zmin,  zmin,  xlim, xlim,  zmin,   zmin,  xlim2, xlim2],
        # all 5 faces (excluding bottom)
        # i=[0, 0, 4, 4, 2, 7, 1, 6, 7, 0],
        # j=[1, 2, 5, 6, 3, 6, 2, 5, 3, 4],
        # k=[2, 3, 6, 7, 7, 2, 6, 1, 0, 7],
        # left, right and top faces
        i=[2, 7, 1, 6, 7, 0],
        j=[3, 6, 2, 5, 3, 4],
        k=[7, 2, 6, 1, 0, 7],
        opacity=0.4
    )]

def boxfrustumEdges():
    zmin = 0
    #     0      1      2     3      4       5       6      7
    x=[xlim, -xlim, -xlim, xlim, xlim2, -xlim2, -xlim2, xlim2]
    y=[ylim,  ylim,  ylim, ylim, ylim2,  ylim2,  ylim2, ylim2]
    z=[zmin,  zmin,  xlim, xlim,  zmin,   zmin,  xlim2, xlim2]
    i=[1,2,3,0,3,7,4,7,6,2,6,5]

    datax = [x[n] for n in i]
    datay = [y[n] for n in i]
    dataz = [z[n] for n in i]

    return [go.Scatter3d(
        x=datax,
        y=datay,
        z=dataz,
        line=dict(color='black', width=2, showscale=False),
        mode='lines',
        name='box edges'
    )]

levelThresh = radians(10)
def rollColorName(roll):
    absroll = abs(roll)
    if absroll < levelThresh:
        # level
        return 'green'
    elif abs(absroll-radians(180)) < levelThresh:
        # inverted
        return 'blue'
    elif abs(absroll-radians(90)) < levelThresh:
        # knife edge
        return 'yellow'
    else:
        return 'red'

green = [0., 1., 0.]
blue = [0., 0., 1.]
yellow = [.8, .8, 0.]
red = [1., 0., 0.]

def rollColor(roll):
    absroll = abs(roll)
    if absroll < levelThresh:
        # level
        return green
    elif abs(absroll-radians(180)) < levelThresh:
        # inverted
        return blue
    elif abs(absroll-radians(90)) < levelThresh:
        # knife edge
        return yellow
    else:
        return red


def meshes(obj, npoints, seq, colour, enu2ned):
    start = seq.data.index[0]
    end = seq.data.index[-1]
    state = [ seq.get_state_from_time(start + (end-start) * i / npoints)
             for i in range(0, npoints+1) ]
    return [
        obj.transform(state[i].transform).create_mesh(
            rollColorName(enu2ned.quat(state[i].att).to_euler().x),
            "{:.1f}".format(start + (end-start) * i / npoints))
        for i in range(0, npoints+1)
    ]

# create a mesh for a "ribbon" plot
# 3 triangles for each pair of poses: current origin to each current/next wingtip
# and origin to next left/right wingtip
def ribbon(scale, seq, enu2ned):
    left  = Point(0, -scale/2, 0)
    right = Point(0,  scale/2, 0)

    # transform origin and wingtips to world frame
    curPose = seq.get_state_from_index(0).transform
    ctr = seq.get_state_from_index(0).pos
    curLeft = curPose.point(left)
    curRight = curPose.point(right)

    # init vertex and face lists
    x = [ctr.x, curLeft.x, curRight.x]
    y = [ctr.y, curLeft.y, curRight.y]
    z = [ctr.z, curLeft.z, curRight.z]
    faces = []
    facecolor = rollColor(enu2ned.quat(seq.get_state_from_index(0).att).to_euler().x)
    facecolors = [facecolor, facecolor, facecolor]

    ctrIndex = 0
    for i in range(1, seq.data.shape[0]):
        # transform origin and wingtips to world frame
        nextPose = seq.get_state_from_index(i).transform
        nextctr = seq.get_state_from_index(i).pos
        nextLeft = nextPose.point(left)
        nextRight = nextPose.point(right)

        # update vertex and face lists
        x.extend([nextctr.x, nextLeft.x, nextRight.x])
        y.extend([nextctr.y, nextLeft.y, nextRight.y])
        z.extend([nextctr.z, nextLeft.z, nextRight.z])

        facecolor = rollColor(enu2ned.quat(seq.get_state_from_index(i).att).to_euler().x)

        # clockwise winding direction
        faces.append([ctrIndex, ctrIndex+1, ctrIndex+4])
        facecolors.append(facecolor)
        faces.append([ctrIndex, ctrIndex+5, ctrIndex+2])
        facecolors.append(facecolor)
        faces.append([ctrIndex, ctrIndex+4, ctrIndex+5])
        facecolors.append(facecolor)

        ctrIndex += 3;

    I, J, K = np.array(faces).T
    return [go.Mesh3d(
        name='ribbon',
        x=x, y=y, z=z, i=I, j=J, k=K,
        intensitymode="cell",
        facecolor=facecolors,
        showlegend=True,
        hoverinfo="none"
    )]

def trace3d(datax, datay, dataz, name, colour='black', width=2, text=None):
    return go.Scatter3d(
        x=datax,
        y=datay,
        z=dataz,
        line=dict(color=colour, width=width),
        mode='lines',
        text=text,
        hoverinfo="text",
        name=name
    )


def cgtrace(seq, name="cgtrace"):
    return trace3d(
        *seq.pos.to_numpy().T,
        colour="black",
        text=["{:.1f}".format(val) for val in seq.data.index],
        name=name
    )


def tiptrace(seq, span, enu2ned):
    def rpyd(i):
        return enu2ned.quat(seq.get_state_from_index(i).att).to_euler() * 180/np.pi
    text = ["t:{:.1f}, roll: {:.1f}, pitch: {:.1f}, yaw: {:.1f}".format(
        seq.data.index[i], rpyd(i).x, rpyd(i).y, rpyd(i).z)
            for i in range(seq.data.shape[0])]

    def make_offset_trace(pos, name, colour, text):
        return trace3d(
            *seq.body_to_world(pos).data.T,
            name=name,
            colour=colour,
            text=text,
            width=1
        )

    return [
        make_offset_trace(Point(0, span/2, 0), "starboard", "green", text),
        make_offset_trace(Point(0, -span/2, 0), "port", "red", text)
    ]


def create_3d_plot(traces):
    return go.Figure(
        traces,
        layout=go.Layout(template="flight3d+judge_view"))

def _axistrace(cid):
    return trace3d(*cid.get_plot_df(20).to_numpy().T)

def axestrace(cids: Union[Coord, List[Coord]]):
    if isinstance(cids, List):
        return [_axistrace(cid) for cid in cids]
    elif isinstance(cids, Coord):
        return _axistrace(cids)
