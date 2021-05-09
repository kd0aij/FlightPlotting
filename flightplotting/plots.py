import plotly.graph_objects as go
import flightplotting.templates
from flightplotting.traces import tiptrace, meshes


def plotsec(sec, obj, scale=10, nmodels=20, fig=None, color="orange"):
    traces = tiptrace(sec, scale * 1.85) + \
        meshes(obj.scale(scale), nmodels, sec, color)

    if fig is None:
        fig = go.Figure(
            data=traces,
            layout=go.Layout(template="flight3d+judge_view")
        )
    else:
        for trace in traces:
            fig.add_trace(trace)
    return fig


def plotdtw(flown, segments):
    fig = go.Figure()

    traces = tiptrace(flown, 10)

    for segname in segments["element"].unique():
        seg = segments[segments.element == segname]
        traces.append(go.Scatter3d(x=seg.x, y=seg.y, z=seg.z,
                               mode='lines', line=dict(width=6), name=segname))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(template="flight3d+judge_view")
    )

    return fig



def create_3d_plot(traces):
    return go.Figure(
        traces,
        layout=go.Layout(template="flight3d+judge_view"))
