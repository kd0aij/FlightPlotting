import plotly.graph_objects as go
from components.model import create_mesh
from geometry import Point

def plot2d(datax, datay, colour='black', width=1, fig=None, name=None, text=None):
    if not fig:
        fig=go.Figure()
    fig.add_trace(go.Scatter(x=datax, y=datay, name=name, text=text, line=dict(color=colour, width=width)))
    fig.update_layout(width = 800, height = 500, margin=dict(l=20, r=20, t=20, b=20),)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1)
    return fig

def plot3d(datax, datay, dataz, colour='black', width=2):
    return go.Scatter3d(
            x=datax, 
            y=datay, 
            z=dataz, 
            line=dict(color=colour, width=width, showscale=True), 
            mode = 'lines'
        )


def meshes(npoints, scale, seq):
    start = seq.data.index[0]
    end = seq.data.index[-1]
    return [create_mesh(seq.get_state_from_time(
            start + (end-start) * i / npoints
        ).transform,
            scale
        ) for i in range(0,npoints)]
        

def trace(seq):
    return go.Scatter3d(x=seq.x, y=seq.y, z=seq.z, mode='lines')

def tiptrace(seq, span):
    return [
        plot3d(*seq.body_to_world(Point(0,span/2,0)).data.T, colour="red"),
        plot3d(*seq.body_to_world(Point(0,-span/2,0)).data.T, colour="blue")
        ]