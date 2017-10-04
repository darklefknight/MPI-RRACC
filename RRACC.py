"""
Created by Tobias Machnitzki (tobias.machnitzki@mpimet.mpg.de) on the 27.10.2017

RRACC stands for "Radar Rain and Cloud Classification"

"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from  matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from datetime import datetime as dt
from calendar import timegm
import argparse
import sys
from joblib import Parallel, delayed


# ----------------------------------------------------------------------------------------------------------------------
# =========================================
# Radar Class
# =========================================
class RadarClass:
    """
    Class for working with Radar data. \n
    For initialisation it needs the path of a netCDF-file with Radar-data. \n

    functions: \n
        Internal:\n
        - __init__:          initializes the class. :param nc_FilePath: Path of a netCDF-file containing Radar-data\n
        - __time2obj:        converts epoche-time to datetime-objects\n
        - __getCloudMask:    calculates the cloud mask\n
        - __getRainMask:     calculates the rain mask\n
        - below:             just for testing and debugging\n
        - __dt_obj:          converts seconds since epoche to a datetime-object\n
        - __ut_obj:          converts a datetime-obejct to seconds since epoche\n

        External:\n
        - print_nc_infos:    prints information of the netCDF-file \n
        - time:              returns a 1D array containing the time as datetime-objects \n
        - range:             returns a 1D array containing the range-gates in m \n
        - Zf:                returns a 1D array containing the reflectivity in dBz \n
        - MeltHeight:        returns a 1D array containing the height of the melting layer in m \n
        - VEL:               returns a 1D array containing the doppler vertical velocity in m/s \n
        - cloudMask():       returns a 2D array containing a cloud-mask (see __getClodMask for more info) \n
        - rainMask():        returns a 2D array containing a rain-mask (see __getRainMask for more info) \n
        - rainRate():        retunrs a 2D array containing the rain intensity in mm/h \n
    """
    def __init__(self, nc_FilePath:str):
        """
        :param nc_FilePath: Path of a netCDF-file containing Radar-data
        """
        nc = Dataset(nc_FilePath)
        self._time = nc.variables['time'][:].copy()  # time in sec since 1970
        self._range = nc.variables['range'][:].copy()  # range in m
        self._Zf = nc.variables['Zf'][:].copy()  # filtered reflectivity
        self._MeltHeight = nc.variables['MeltHei'][:].copy()  # Height of the meltinglayer in m
        self._VEL = nc.variables['VEL'][:].copy()  # "vertical velocity of all Hydrometeors"
        self.print_nc_infos(nc)
        nc.close()

        self._cloudMask = np.asarray(self._Zf).copy()
        self._rainMask = np.asarray(self._Zf).copy()
        self._rainRate = np.asarray(self._Zf).copy()
        self.Below0C = np.asarray(self._Zf)

        self._obj_time = []

        # functions to be executed on initiate:
        self.__time2obj()  # creates datetime-objects from "time"
        self.__getCloudMask()  # creates a cloudmask
        self.__getRainMask()  # creates a rainmask

    def __time2obj(self):
        for element in self._time:
            self._obj_time.append(dt.replace(self.__dt_obj(element),tzinfo=None))
        self._obj_time = np.asarray(self._obj_time)

    def __getCloudMask(self):
        # TODO: Unterscheidung zwischen Cirrus und Cumulus via LDR and Melting Layer hight

        self._cloudMask[self._cloudMask > -99999] = np.nan # set complete array to nan
        self._cloudMask[np.logical_and(self.VEL() > -1, self._Zf <= 0)] = 10# downdrafts in cloud
        self._cloudMask[np.logical_and((self.Zf() > -50), (self.VEL() > -1))] = 30  # cloud
        self._cloudMask[self.Zf() <= -50] = -50  # cloud-beards

        self._cloudMask[self.VEL() <= -1] = 0  # rain

    def __getRainMask(self):
        self._rainMask[self._rainMask > -99999] = np.nan # set complete array to nan
        self._rainRate[self._rainRate > -99999] = np.nan # set complete array to nan

        # rain where VEL < 1 and Zf > 0:
        self._rainMask[np.logical_and(self.VEL() <= -1,self._Zf > 0)] = self._Zf[np.logical_and(self.VEL() <= -1,self._Zf > 0)]  # rain

        # Precipitation only below melting-layer:

        self.Below0C = [self.Below0C == -99999][0]
        for i in range(len(self._time)):
            self.Below0C[i][np.where(self._range < self._MeltHeight[i])] = True

        self._rainMask[~self.Below0C] = np.nan

        # Marshall-Palmer z-R relationship:
        # RR = 0.036 * 10^(0.0625 * dBZ)
        self._rainRate = np.multiply(0.036, np.power(10,np.multiply(0.0625 ,self._rainMask)))

        # throw away to small values
        self._rainRate[self._rainRate < 0.1] = np.nan

    def below(self):
        return self.Below0C

    def print_nc_infos(self, nc):
        print(nc)
        print('---------------------------------------')
        for key in nc.variables.keys():
            print(key)

    def __dt_obj(self, u):
        return dt.utcfromtimestamp(u)

    def __ut_obj(self, d):
        return timegm(d.timetuple())

    def time(self, shape:str="dt"):
        """
        :param shape: what shape the returned values should have. Allowed: \n
                "dt" = datetime-object \n
                "ut" = seconds since 1970
        :return: datetime-obj or seconds since 1970, depending on parameter-settings
        """

        if shape == "dt":
            return self._obj_time

        elif shape == "ut":
            return self._time

    def range(self):
        return self._range

    def Zf(self):
        return self._Zf

    def MeltHeight(self):
        return self._MeltHeight

    def VEL(self):
        return self._VEL

    def cloudMask(self):
        return self._cloudMask

    def rainMask(self):
        return self._rainMask

    def rainRate(self):
        return self._rainRate


# ----------------------------------------------------------------------------------------------------------------------

def contourf_plot(MBR2, value):
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cf1 = ax1.contourf(MBR2.time(), MBR2.range(), MBR2.cloudMask().transpose(), cmap="jet", label="Cloudmask")
    ax1.plot(MBR2.time(),MBR2.MeltHeight(),color='black',ls="--",label="Melting layer hight")
    ax1.legend(loc="best")
    ax1.set_ylim(0, 14000)
    fig.colorbar(cf1,ax=ax1, label="[dBZ]")
    ax1.set_title("Cloudmask")

    cf2 = ax2.contourf(MBR2.time(), MBR2.range(), value.transpose(), cmap="jet", label="Rain rate")
    ax2.plot(MBR2.time(),MBR2.MeltHeight(),color='black',ls="--",label="Melting layer hight")
    ax2.set_ylim(0, 14000)
    fig.colorbar(cf2,ax=ax2,label="[mm/h]")
    ax2.legend(loc="best")
    ax2.set_title("Precipitation rate")

# ----------------------------------------------------------------------------------------------------------------------

def getColorFromValue(min, max, array, cmap="jet"):
    """
    Converts a 2D valaue array to rgb-colors.

    :param min: minimum of the array
    :param max: maximum of the array
    :param array: 2D array
    :param cmap: string of the colormap to use

    :return: 3D color-array
    """
    norm = Normalize(vmin=min,vmax=max)
    relative_value = norm(array)
    color_map = get_cmap(cmap)
    color = color_map(relative_value)
    print(np.shape(color))
    color[:,:,0:3] *= 255
    return color





def getColormapAsList(steps=100,cmap="jet"):
    def clamp(x):
        return int(max(0, min(x, 255)))

    values = np.linspace(0,1,steps)
    color_map = get_cmap(cmap)

    colors = []
    for value in values:
        color = color_map(value)
        colors.append("#{0:02x}{1:02x}{2:02x}".format(clamp(color[0]*255), clamp(color[1]*255), clamp(color[2]*255)))

    return colors




if __name__ == "__main__":

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

    contourf_plot(Radar, Radar.rainRate())
