from flightdata.fields import Field
import streamlit as st

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from flightanalysis.section import Section
from flightanalysis.flightline import FlightLine
from flightdata import Flight, Fields
from components.plots import meshes, trace, tiptrace


bin = Flight.from_csv('/home/tom/notebooks/AutoJudge/schedules/136.csv')


@st.cache  # TODO this may not notice changes to submodules
def load_data():
    flight = bin.subset(101, 490)
    return flight, Section.from_flight(flight, FlightLine.from_covariance(flight))


flight, seq = load_data()



npoints = st.sidebar.number_input("Number of Models", 0, 50, value=20)
scale = st.sidebar.number_input("Model Scale Factor", 1.0, 50.0, value=10.0)
showmesh = st.sidebar.checkbox("Show Models", False)
cgtrace = st.sidebar.checkbox("Show CG Trace", True)
ttrace = st.sidebar.checkbox("Show Tip Trace", False)


plot_range = st.slider(
    "plot range", 0.0, flight.duration, (0.0, flight.duration))

def make_plot_data():
    sec = seq.subset(*plot_range)
    traces = []
    if showmesh:
        traces += [mesh for mesh in meshes(npoints, scale, sec)]
    if cgtrace:
        traces += [trace(sec)]
    if ttrace:
        traces += tiptrace(sec, scale * 1.85)
    return traces

st.plotly_chart(
    go.Figure(
        make_plot_data(),
        layout=go.Layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(aspectmode='data')
        )),
    use_container_width=True
)
