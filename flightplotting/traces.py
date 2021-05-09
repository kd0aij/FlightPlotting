import plotly.graph_objects as go
import flightplotting.templates
from .model import OBJ
from geometry import Point, Coord
import numpy as np
from typing import List, Union
from math import cos, sin, tan, radians


def boxtrace():
    xlim=170*tan(radians(60))
    ylim=170
    return [go.Mesh3d(
        #  0  1     2     3      4    5      6
        x=[0, xlim, 0,    -xlim, xlim, 0,   -xlim], 
        y=[0, ylim, ylim,  ylim, ylim, ylim, ylim], 
        z=[0, 0,    0,     0,    xlim, xlim, xlim], 
        i=[0, 0, 0, 0, 0], 
        j=[1, 2, 1, 3, 4], 
        k=[2, 3, 4, 6, 6],
        opacity=0.4
    )]


def meshes(obj, npoints, seq, colour):
    start = seq.data.index[0]
    end = seq.data.index[-1]
    return [
        obj.transform(
            seq.get_state_from_time(
                start + (end-start) * i / npoints
            ).transform
        ).create_mesh(
            colour,
            "{:.1f}".format(start + (end-start) * i / npoints)
        ) for i in range(0, npoints+1)
    ]


def trace3d(datax, datay, dataz, colour='black', width=2, text=None, name="trace3d"):
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



def tiptrace(seq, span):
    text = ["{:.1f}".format(val) for val in seq.data.index]

    def make_offset_trace(pos, colour, text):
        tr =  trace3d(
            *seq.body_to_world(pos).data.T,
            colour=colour,
            text=text,
            width=1
        )
        tr['showlegend'] = False
        return tr

    return [
        make_offset_trace(Point(0, span/2, 0), "blue", text),
        make_offset_trace(Point(0, -span/2, 0), "red", text)
    ]


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
