import plotly.graph_objects as go
from .model import OBJ
from geometry import Point, Coord
import numpy as np
from typing import List, Union
from math import cos, sin, tan, radians


xlim=170*tan(radians(60))
ylim=170


def boxtrace():
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


def meshes(obj, npoints, seq):
    start = seq.data.index[0]
    end = seq.data.index[-1]
    return [
        obj.transform(
            seq.get_state_from_time(
                start + (end-start) * i / npoints
            ).transform
        ).create_mesh(
            "{:.1f}".format(start + (end-start) * i / npoints)
        ) for i in range(0, npoints+1)
    ]


def trace3d(datax, datay, dataz, colour='black', width=2, text=None):
    return go.Scatter3d(
        x=datax,
        y=datay,
        z=dataz,
        line=dict(color=colour, width=width, showscale=True),
        mode='lines',
        text=text,
        hoverinfo="text"
    )


def trace(seq):
    return trace3d(
        *seq.pos.to_numpy().T,
        colour="black",
        text=["{:.1f}".format(val) for val in seq.data.index]
    )


def tiptrace(seq, span):
    text = ["{:.1f}".format(val) for val in seq.data.index]

    def make_offset_trace(pos, colour, text):
        return trace3d(
            *seq.body_to_world(pos).data.T,
            colour=colour,
            text=text
        )

    return [
        make_offset_trace(Point(0, span/2, 0), "blue", text),
        make_offset_trace(Point(0, -span/2, 0), "red", text)
    ]


def create_3d_plot(traces):
    return go.Figure(
        traces,
        layout=go.Layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(aspectmode='data')
        ))

def _axistrace(cid):
    return trace3d(*cid.get_plot_df(20).to_numpy().T)

def axestrace(cids: Union[Coord, List[Coord]]):
    if isinstance(cids, List):
        return [_axistrace(cid) for cid in cids]
    elif isinstance(cids, Coord):
        return _axistrace(cids)
