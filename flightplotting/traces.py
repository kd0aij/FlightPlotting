import plotly.graph_objects as go
import flightplotting.templates
from flightanalysis import State

from .model import OBJ
from geometry import Point, Coord, Quaternion, Transformation
from geometry.point import vector_norm, normalize_vector, dot_product, cross_product
import numpy as np
from typing import List, Union
from math import cos, sin, tan, radians, asin, copysign, sqrt, pi


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

levelThresh = radians(15)
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

def wrapPi(r, hyst=0):
    while r > pi + hyst:
        r -= 2*pi
    while r < -180 - hyst:
        r += 2*pi
    return r

def getManeuverPlane(rhdg, ghdg):
    # constrain maneuver hdg to box or cross-box
    revhdg = wrapPi(rhdg + pi)
    chdg = wrapPi(rhdg + pi/2)
    if abs(ghdg - rhdg) < pi/4:
      mhdg = rhdg;
    elif abs(ghdg - revhdg) < pi/4:
      mhdg = revhdg;
    elif abs(ghdg - chdg) < pi/4:
      mhdg = chdg;
    else:
      mhdg = wrapPi(chdg + pi)

    return mhdg

# generate maneuver RPY for each element of a Section
def genManeuverRPY(seq, rhdg, mingspd, pThresh, enu2ned):
    N = seq.data.shape[0]
    roll = np.empty(N)
    pitch = np.empty(N)
    wca = np.empty(N)
    xwnd = np.empty(N)
    wca_axis = []
    mhdg = np.empty(N)
    
    
    # calculate ground heading qualified by a minimum groundspeed: mingspd
    # retain previous heading while speed is below mingspd
    # vel has been rotated into ENU frame; not sure whether it is also rotated to contest frame
    # rhdg = 0
    ghdg = np.zeros(N)
    ghdg[1] = rhdg;
    for i in range(0, N):
      vel = seq.get_state_from_index(i).vel
      # vel = enu2ned.rotate(seq.get_state_from_index(i).vel)
      # try inverse transform
      vel = enu2ned.rotation.inverse().transform_point(seq.get_state_from_index(i).vel)
      spd = sqrt(vel.x**2 + vel.y**2)
      if spd > mingspd:
        ghdg[i] = np.arctan2(vel.y, vel.x)
      else:
        ghdg[i] = ghdg[i-1]
        
        
    onVertical = False
    hyst = np.radians(1)
    mplanes = []
    mplane = {"hdg":0,"pos":Point(0,0,0),"entry":False}

    for i in range(1, N):
        curState = seq.get_state_from_index(i)
        t = seq.data.index[i]
        att = seq.get_state_from_index(i).att
        
        # rotate back to NED
        # att = enu2ned.rotation.inverse() * att
        # att = enu2ned.rotation * att
        
        # rotate about z to world frame runway heading
        zrot = Quaternion.from_axis_angle(Point(0, 0, rhdg))
        att = zrot * seq.get_state_from_index(i).att
        
        e_pitch = eulerPitch(att)
        # determine maneuver heading based on whether this is a vertical line
        if onVertical:
            # maneuver heading is current mplane heading
            mplane["hdg"] = getManeuverPlane(rhdg, ghdg[i])
            mhdg[i] = mplane["hdg"]
            
            # check for exit from vertical line
            if (abs(e_pitch) < (pThresh - hyst)):
              onVertical = 0;
              # on exit from vertical line
              # use ground heading to define maneuver plane
              print("exit from vertical line")
              # set maneuver plane heading to current ground heading
              mplane["hdg"] = ghdg[i]
              mplane["pos"] = curState.pos
              mplane["entry"] = False
              print("t: {:5.1f} pitch: {:3.1f}, maneuver heading: {:3.0f}, ghdg: {:3.0f}".format(
                  t, np.degrees(e_pitch), np.degrees(mplane["hdg"]), np.degrees(ghdg[i])))
              # record ground heading maneuver plane
              mplanes.append(mplane)
              mhdg[i] = ghdg[i];
        else:
            # maneuver heading is just ground heading
            mhdg[i] = ghdg[i];
            
            # entering vertical line if pitch > threshold
            if (abs(e_pitch) > (pThresh)):
              onVertical = 1;
              # on entry to vertical line:
              print("entry to vertical line")
              # pick aerobatic box heading using previous ground heading
              mplane["hdg"] = getManeuverPlane(rhdg, ghdg[i])
              mplane["pos"] = curState.pos
              mplane["entry"] = True
              mhdg[i] = mplane["hdg"];
              print("t: {:5.1f} pitch: {:3.1f}, maneuver heading: {:3.0f}, ghdg: {:3.0f}".format(
                  t, np.degrees(e_pitch), np.degrees(mplane["hdg"]), np.degrees(ghdg[i])))
              # record vertical maneuver plane
              mplanes.append(mplane)

        [roll[i], pitch[i], wca[i], axis] = maneuverRPY(mhdg[i], att)
        # invert roll
        # roll[i] = wrapPi(roll[i] + pi)
        wca_axis.append(axis)
        
        # if abs(wca[i]) > np.radians(12):
        #   print("large wca: {:5.1f}".format(np.degrees(wca[i])))
          
        # crosswind is ~ |vENU|*sin(wca): so percentage of earthframe velocity is:
        xwnd[i] = 100 * abs(np.sin(wca[i]))
        
    return [roll, pitch, wca, axis]


# calculate heading for maneuver plane
# TODO: extend this to find and use maneuver plane only on vertical lines
#       Otherwise use normal Euler fixed angles
def mPlane(chdg, vel3d:Point):
    ghdg = np.arctan2(vel3d.y, vel3d.x)

    # constrain heading to chdg or pi+chdg
    if np.abs(wrapPi(ghdg - chdg)) > pi/2:
        hdg = wrapPi(chdg + pi)
    else:
        hdg = chdg
    return hdg

def eulerPitch(q: Quaternion):
    _sinp = 2 * (q.w * q.y - q.z * q.x)
    if abs(_sinp) >= 1:
        pitch = copysign(pi / 2, _sinp)
    else:
        pitch = asin(_sinp)
        
    return pitch

# rhdg: maneuver heading in radians: contest frame?
# quat: body to contest? rotation
# return: [roll, pitch, wca, wca_axis]
def maneuverRPY(rhdg: float, quat: Quaternion) -> [float, float, float, Point]:
    # given maneuver heading rhdg, calculate roll angle as angle between
    # rhdg/earthz plane and body x/y plane
    bx = quat.transform_point(Point(1, 0, 0));
    # rhdg = mPlane(rhdg, vel3d)

    # hzplane is the normal vector which defines the maneuver plane 
    # this hzplane requires maneuvers to lie in a vertical plane parallel to rhdg
    hzplane = Point(-sin(rhdg), cos(rhdg), 0);

  # a more general version would allow the maneuver plane to be non-vertical
  # where mplane is (hv cross earthz) rotated about hv by a roll angle
##  hv = Point(cosd(rhdg), sind(rhdg) 0);
##  hzplane = cross_product(hv, mplane);

    # the wind correction angle (WCA) relative to flight path is the
    # angle between body frame x and hzplane
    # This should be independent of roll and pitch: roll does not affect the direction
    # of bx and pitch is a rotation about hzplane, which does not change the angle
    wca_axis = cross_product(bx, hzplane);
    wca = (pi/2) - np.arctan2(vector_norm(wca_axis), dot_product(bx, hzplane));

    # to back out wca, rotate about cross(bx, hzplane)
    wca_axis = normalize_vector(wca_axis)
    r2hzp = Quaternion.from_axis_angle(wca_axis * -wca)

    # this is the attitude with body x rotated into maneuver plane
    fq = (r2hzp * quat).norm()

    # calculate Euler pitch in maneuver plane
    rpy = fq.to_euler()
    pitch = rpy.y

    # HACK: reverse rhdg if sign of euler yaw is different from that of rhdg
    # this is detecting a reversal in ground course at low gspd, but I had thought
    # that flipping the hzplane normal shouldn't affect the results
    if np.sign(rpy.z) != np.sign(rhdg):
        rhdg = wrapPi(rhdg + pi)

    # back out rhdg and pitch
    ryaw = Quaternion.from_axis_angle(Point(0, 0, 1) * -rhdg)
    rpitch = Quaternion.from_axis_angle(Point(0, 1, 0) * -pitch)

    # remaining rotation should be roll relative to maneuver plane
    rollq = (rpitch * ryaw * fq).norm()
    axisr = Quaternion.to_axis_angle(rollq)
    thetar = abs(axisr)
    axisr /= thetar
    direction = dot_product(axisr, Point(1, 0, 0))
    # invert roll relative to Octave implementation
    roll = np.sign(direction) * wrapPi(thetar + pi)

    return [roll, pitch, wca, wca_axis]

def meshes(obj, npoints, seq, colour, enu2ned):
    start = seq.data.index[0]
    end = seq.data.index[-1]
    state = [ seq.get_state_from_time(start + (end-start) * i / npoints)
             for i in range(0, npoints+1) ]
    return [
        obj.transform(state[i].transform).create_mesh(
            rollColorName(maneuverRPY(0, state[i].att)[0]),
            "{:.1f}".format(start + (end-start) * i / npoints))
        for i in range(0, npoints+1)
    ]

# create a mesh for a "ribbon" plot
# 3 triangles for each pair of poses: current origin to each current/next wingtip
# and origin to next left/right wingtip
def ribbon(scale, seq, roll):
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
    facecolors = []

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

        facecolor = rollColor(roll[i])

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


def manoeuvretraces(seq):
    traces = []
    for name, manoeuvre in seq.split_manoeuvres().items():
        traces.append(go.Scatter3d(
            x=manoeuvre.x,
            y=manoeuvre.y,
            z=manoeuvre.z,
            mode='lines',
            text=manoeuvre.element,
            hoverinfo="text",
            name=name
        ))

    return traces


def elementtraces(seq):
    traces = []
    for name, element in seq.split_elements().items():
        traces.append(go.Scatter3d(
            x=element.x,
            y=element.y,
            z=element.z,
            mode='lines',
            text=element.manoeuvre,
            hoverinfo="text",
            name=name
        ))

    return traces


def tiptrace(seq, span, roll, pitch, wca):
    def rpyd(i):
        # [roll, pitch, wca, wca_axis] = maneuverRPY(0, seq.get_state_from_index(i).att)
        # return enu2ned.quat(seq.get_state_from_index(i).att).to_euler() * 180/pi
        return Point(roll[i], pitch[i], wca[i]) * 180/pi
    text = ["t:{:.1f}, roll: {:.1f}, pitch: {:.1f}, wca: {:.1f}".format(
        seq.data.index[i], rpyd(i).x, rpyd(i).y, rpyd(i).z)
            for i in range(seq.data.shape[0])]
    # text = ["t:{:.1f}, roll: {:.1f}".format(
    #     seq.data.index[i], roll[i] * 180/pi)
    #         for i in range(seq.data.shape[0])]

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

def axis_rate_trace(sec, ab = False):
    if ab:
        return [
            go.Scatter(x=sec.data.index, y=abs(sec.brvr), name="r"),
            go.Scatter(x=sec.data.index, y=sec.brvp, name="p"),
            go.Scatter(x=sec.data.index, y=abs(sec.brvy), name="y")]
    else:
        return [
            go.Scatter(x=sec.data.index, y=sec.brvr, name="r"),
            go.Scatter(x=sec.data.index, y=sec.brvp, name="p"),
            go.Scatter(x=sec.data.index, y=sec.brvy, name="y")]

def _axistrace(cid):
    return trace3d(*cid.get_plot_df(20).to_numpy().T)

def axestrace(cids: Union[Coord, List[Coord]]):
    if isinstance(cids, List):
        return [_axistrace(cid) for cid in cids]
    elif isinstance(cids, Coord):
        return _axistrace(cids)
