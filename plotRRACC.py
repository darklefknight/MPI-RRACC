# ============Importing ================

from RRACC import RadarClass, getColorFromValue, getColormapAsList
import numpy as np
from bokeh.io import curdoc, show
from bokeh.models import (
    Range1d,
    DatetimeTickFormatter,
    ColumnDataSource,
    CustomJS,
    Legend,
    LinearColorMapper,
    BasicTicker,
    ColorBar,
    PrintfTickFormatter,
    HoverTool)
from bokeh.models.widgets import Select, Button
from bokeh.layouts import gridplot, column, widgetbox, column, row, layout
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.events import ButtonClick, Tap, LODStart
import matplotlib.dates as mdate
from datetime import timedelta
from datetime import datetime as dt
from functools import lru_cache
import os
import argparse
import sys
import pandas as pd

time_resolution = 10  # Time-Resolution of the radar in seconds

# =============Set up data =====================

# Get parsed arguments:
parser = argparse.ArgumentParser(description="example: RRACC.py 20170401", prog="RRACC.py")
parser.add_argument('datestr', metavar='t', help='add date-argument YYYYMMDD', type=int)
parser.add_argument('--device', dest='device', help='specify a device. Allowed: MBR2, KATRIN. Default: MBR2',
                    default="MBR2")
args = parser.parse_args()
datestr = str(args.datestr)
print(datestr)
datestr = datestr[2:]

# Set nc-file depending on parsed argument:
DEVICE = str(args.device)
if DEVICE == "MBR2":
    NC_PATH = "/data/mpi/mpiaes/obs/m300517/RadarCorrection/MBR2_out/"
    NC_FILE = "MMCR__MBR__Spectral_Moments__10s__155m-25km__" + datestr + ".nc"

elif DEVICE == "KATRIN":
    NC_PATH = "/data/mpi/mpiaes/obs/m300517/RadarCorrection/KATRIN_out/"
    NC_FILE = "MMCR__KATRIN__Spectral_Moments__10s__300m-15km__" + datestr + ".nc"

else:
    print("InputError: --device can be either MBR2 or KATRIN")
    sys.exit(1)

# Initiate class:
Radar = RadarClass(NC_PATH + NC_FILE)

Toolbox = "save,box_zoom,pan,wheel_zoom,undo"
p1 = figure(title="Radar Rain and Cloud Classification", tools=Toolbox, responsive=True, x_axis_type='datetime',
            y_range=Range1d(0, Radar.range()[-1], bounds="auto"),
            x_range=Range1d(Radar.time()[0], Radar.time()[-1], bounds="auto"),
            )  # set up the first plot

meltsource = ColumnDataSource(
    data=dict(
        x=Radar.time(),
        y=Radar.MeltHeight(),
    )
)
p1.line(x="x", y="y", source=meltsource, line_color="black", line_dash="dashed", line_width=2, legend="Melting Layer Height")

contourlist = []
colorbarlist = []
def contourfPlot(Radar_var,name, minmax=None):
    # ========== get Colors =====================
    if minmax == None:
        max_dbz = np.nanmax(Radar_var)
        min_dbz = np.nanmin(Radar_var)
    else:
        min_dbz,max_dbz = minmax


    rangeGateHeight = Radar.range()[2] - Radar.range()[1]
    rgb = getColormapAsList()

    # ========== prepare data for plotting ==============
    df = pd.DataFrame(Radar_var, index=Radar.time(), columns=Radar.range())
    df.columns.name = "range"
    df.index.name = "time"
    df1 = pd.DataFrame(df.stack(), columns=['data']).reset_index()

    source = ColumnDataSource(df1)

    mapper = LinearColorMapper(palette=rgb, low=min_dbz, high=max_dbz)
    # ====================================


    cf = p1.rect(x="time", y="range", height=rangeGateHeight, width=timedelta(seconds=time_resolution), source=source,
                 fill_color={'field': 'data', 'transform': mapper}, legend=name )
    cf.glyph.line_color = cf.glyph.fill_color

    contourlist.append(cf)
    p1.plot_width = 1300
    p1.plot_height = 1000
    p1.sizing_mode = "scale_width"

    # Colorbar:
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                         ticker=BasicTicker(desired_num_ticks=int(len(rgb) / 2)),
                         formatter=PrintfTickFormatter(format="%d"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    colorbarlist.append(color_bar)


contourfPlot(Radar.cloudMask(),"cloudMask")
contourfPlot(Radar.rainRate(),"rainRate",minmax=(0,10))

colorbarlist[0].title = "Cloudmask-value"
colorbarlist[1].title = "Rain rate [mm/h]"


p1.add_layout(colorbarlist[0],"left")
p1.add_layout(colorbarlist[1],"right")
contourlist[0].glyph.fill_alpha = 0.2
contourlist[0].glyph.line_alpha = 0

# Hover-tool:
hover = HoverTool(
    tooltips=[
        ('time', '@time{%H:%M:%S}'),
        ('value', '@data dBZ')
    ],
    formatters={
        'time': 'datetime'
    }
)
p1.add_tools(hover)
grid = gridplot([[p1]])

show(grid)

#TODO: Add buttons to switch between plots