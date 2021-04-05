#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 16:16:44 2021

@author: markw
"""

from flightanalysis import Section, FlightLine
from flightdata import Flight
import numpy as np
from flightplotting.plots import tiptrace, meshes, ribbon, create_3d_plot
from flightplotting.model import OBJ
from geometry import Transformation, Point, Quaternion, Points

obj = OBJ.from_obj_file('../../data/models/ColdDraftF3APlane.obj').transform(
    Transformation(
        Point(0.75, 0, 0),
        Quaternion.from_euler(Point(np.pi, 0, -np.pi/2))
    )
)

binfile = "P21_032521"
bin = Flight.from_log('../../data/logs/' + binfile + ".BIN")

sec = Section.from_flight(bin, FlightLine.from_covariance(bin))

start = 268
end = 303
subSec = sec.subset(start, end)

fig = create_3d_plot(tiptrace(subSec, 10*1.85) + ribbon(9, subSec) + meshes(obj.scale(10), 10, subSec))

# save interactive figure
fname = "%s_%s-%s.html" % (binfile, start, end)
fig.write_html(fname)

# create and save 3-views
import plotly.io as pio
pio.renderers.default = 'svg'

name = 'eye = (x:2.5, y:0., z:0.)'
camera = dict(
    eye=dict(x=2.5, y=0, z=0)
)
fig.update_layout(scene_camera=camera, title=name)
fig.write_image("images/Xview.svg")
fig.show()

fig.update_layout(scene_camera=camera, title=name)
name = 'eye = (x:0., y:2.5., z:0.)'
camera = dict(
    eye=dict(x=0., y=2.5, z=0.)
)
fig.update_layout(scene_camera=camera, title=name)
fig.write_image("images/Yview.svg")
fig.show()

fig.update_layout(scene_camera=camera, title=name)
name = 'eye = (x:0., y:0., z:2.5)'
camera = dict(
    eye=dict(x=0., y=0, z=2.5)
)
fig.update_layout(scene_camera=camera, title=name)
fig.write_image("images/Zview.svg")
fig.show()
