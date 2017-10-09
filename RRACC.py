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
from scipy.ndimage.morphology import binary_closing,binary_opening
from scipy.sparse.csgraph import connected_components
from scipy import ndimage

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
        self._BCOtime = nc.variables["bco_day"][:].copy()
        self._range = nc.variables['range'][:].copy()  # range in m
        self._Zf = nc.variables['Zf'][:].copy()  # filtered reflectivity
        self._MeltHeight = nc.variables['MeltHei'][:].copy()  # Height of the meltinglayer in m
        self._VEL = nc.variables['VEL'][:].copy()  # vertical velocity of all hydrometeors
        self._VELg = nc.variables['VELg'][:].copy() # Doppler velocity of all targets
        self._LDR = nc.variables['LDR'][:].copy() # Linear depolarization rate of all Hydrometeors
        self._LDRg = nc.variables['LDRg'][:].copy() # Linear depolarization rate of all targets
        self.print_nc_infos(nc)
        nc.close()

        self._cloudMask = np.asarray(self._Zf).copy()
        self._rainMask = np.asarray(self._Zf).copy()
        self._rainRate = np.asarray(self._Zf).copy()
        self._notSphericMask = np.asarray(self._VELg).copy()

        self._obj_time = []

        # functions to be executed on initiation:
        self.__time2obj()  # creates datetime-objects from "time"
        self.__getCloudMask()  # creates a cloudmask
        self.__getRainMask()  # creates a rainmask
        self.__getNotSpheric() #gets a mask for all not spheric particles


    def __time2obj(self):
        for element in self._time:
            self._obj_time.append(dt.replace(self.__dt_obj(element),tzinfo=None))
        self._obj_time = np.asarray(self._obj_time)

    def __getNotSpheric(self):
        self._notSphericMask[self._notSphericMask > -99999] = False
        # self._notSphericMask[]

    def __getCloudMask(self):
        """
        Creates a cloud mask.
        Values stand for:
        0 = cloud-beard
        1 = rain
        2 = cloud above melting layer
        3 = cloud below melting layer

        :return:
        """
        # TODO: Unterscheidung zwischen Cirrus und Cumulus via LDR and Melting Layer hight

        self._cloudMask[self._cloudMask > -99999] = np.nan # set complete array to nan
        # self._cloudMask[np.logical_and(self.VEL() < -1, self._Zf <= 0)] = 0# downdraft in cloud
        self._cloudMask[np.logical_and((self.Zf() > -50), (self.VEL() > -1))] = 30  # cloud
        self._cloudMask[self.Zf() <= -50] = 0  # cloud-beards

        self._cloudMask[self.VEL() <= -1] = 1  # rain

        __Below0CloudMask = np.asarray(self._cloudMask).copy()
        __Below0CloudMask = [__Below0CloudMask == -99999][0]
        __Below0CloudBeard = np.asarray(self._cloudMask).copy()
        __Below0CloudBeard = [__Below0CloudBeard == -99999][0]


        for i in range(len(self._time)):
            __Below0CloudMask[i][np.where(self._range < self._MeltHeight[i])] = True
            __Below0CloudBeard[i][np.where(self._range < self._MeltHeight[i])] = True

        self._cloudMask[np.logical_and(~__Below0CloudMask,self._cloudMask==30)] = 2 # everything above Melting layer height is cirrus
        self._cloudMask[np.logical_and(__Below0CloudMask,self._cloudMask==30)] = 3  # everything below Melting layer height is cumulus
        self._cloudMask[np.logical_and(~__Below0CloudBeard,self._cloudMask==0)] = np.nan  # cloud-beards just occur below melting layer height
        del __Below0CloudMask, __Below0CloudBeard



    def __getRainMask(self):
        self._rainMask[self._rainMask > -99999] = np.nan # set complete array to nan
        self._rainRate[self._rainRate > -99999] = np.nan # set complete array to nan

        # rain where VEL < 1 and Zf > 0:
        self._rainMask[np.logical_and(self.VEL() <= -1,self._Zf > 0)] = self._Zf[np.logical_and(self.VEL() <= -1,self._Zf > 0)]  # rain

        # Precipitation only below melting-layer:
        __Below0C = np.asarray(self._rainMask)
        __Below0C = [__Below0C == -99999][0]
        for i in range(len(self._time)):
            __Below0C[i][np.where(self._range < self._MeltHeight[i])] = True

        self._rainMask[~__Below0C] = np.nan
        del __Below0C

        # Marshall-Palmer z-R relationship:
        # RR = 0.036 * 10^(0.0625 * dBZ)
        self._rainRate = np.multiply(0.036, np.power(10,np.multiply(0.0625 ,self._rainMask)))

        # throw away to small values
        self._rainRate[self._rainRate < 0.1] = np.nan

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

    def LDR(self):
        return self._LDR

    def LDRg(self):
        return self._LDRg

    def BCOtime(self):
        return self._BCOtime



# ----------------------------------------------------------------------------------------------------------------------

def contourf_plot(MBR2, value):
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)
    ax3 = fig.add_subplot(311)

    cf3 = ax3.contourf(MBR2.time(), MBR2.range(), MBR2.Zf().transpose(), cmap="jet", label="Reflectivity")
    ax3.plot(MBR2.time(), MBR2.MeltHeight(), color='black', ls="--", label="Melting layer hight")
    ax3.legend(loc="best")
    ax3.set_ylim(0, 20000)
    fig.colorbar(cf3, ax=ax3, label="[dBZ]")
    ax3.set_title("Reflectivity")

    cf1 = ax1.contourf(MBR2.time(), MBR2.range(), MBR2.cloudMask().transpose(), cmap="jet", label="Cloudmask")
    ax1.plot(MBR2.time(),MBR2.MeltHeight(),color='black',ls="--",label="Melting layer hight")
    ax1.legend(loc="best")
    ax1.set_ylim(0, 20000)
    cb1 = fig.colorbar(cf1,ax=ax1, label="CloudMaskValue")
    ax1.set_title("Cloudmask")

    # labels = [item.get_text() for item in cb1.get_ticklabels()]
    cb1.set_clim(0,3)
    cb1.set_ticks([0,1,1.8,3])
    cb1.set_ticklabels(["Cloudbeard","Rain","Cirrus","Cumulus"])


    cf2 = ax2.contourf(MBR2.time(), MBR2.range(), value.transpose(), cmap="jet", label="Rain rate")
    ax2.plot(MBR2.time(),MBR2.MeltHeight(),color='black',ls="--",label="Melting layer hight")
    ax2.set_ylim(0, 20000)
    fig.colorbar(cf2,ax=ax2,label="[mm/h]")
    ax2.legend(loc="best")
    ax2.set_title("Precipitation rate")

def plotCloudmask(MBR2,value):
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(212)
    ax2 = fig.add_subplot(211)

    cf1 = ax1.contourf(MBR2.time(), MBR2.range(), value.transpose(), cmap="jet", label="Cloudmask")
    ax1.plot(MBR2.time(),MBR2.MeltHeight(),color='black',ls="--",label="Melting layer hight")
    ax1.legend(loc="best")
    ax1.set_ylim(0, 20000)
    cb1 = fig.colorbar(cf1,ax=ax1, label="Cloud classification")
    ax1.set_title("Cloudmask")

    # labels = [item.get_text() for item in cb1.get_ticklabels()]
    cb1.set_clim(0,3)
    cb1.set_ticks([0,1,1.8,3])
    cb1.set_ticklabels(["Cloudbeard","Rain","Cirrus","Cumulus"])

    cf2 = ax2.contourf(MBR2.time(), MBR2.range(), MBR2.cloudMask().transpose(), cmap="jet", label="Cloudmask")
    ax2.plot(MBR2.time(),MBR2.MeltHeight(),color='black',ls="--",label="Melting layer hight")
    ax2.legend(loc="best")
    ax2.set_ylim(0, 20000)
    cb2 = fig.colorbar(cf2,ax=ax2, label="Cloud classification")
    ax2.set_title("Cloudmask")

    # labels = [item.get_text() for item in cb1.get_ticklabels()]
    cb2.set_clim(0,3)
    cb2.set_ticks([0,1,1.8,3])
    cb2.set_ticklabels(["Cloudbeard","Rain","Cirrus","Cumulus"])

    plt.savefig('Cloudmask.png')

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

def getValuesInArray(array):
    value_list = []
    for i in range(len(array)):
        for element in array[i]:
            if (not element in value_list) and (not np.isnan(element)):
                value_list.append(element)

    return value_list


def create_netCDF(Radar,NC_FILE,nc_name, path_name=''):
    """

    :param nc_name: Name of the netCDF4 file
    :param path_name: Where the netCDF4 file will be written

    """
    from netCDF4 import Dataset
    import time
    import os

    MISSING_VALUE = -999

    nc = Dataset(path_name + nc_name, mode='w', format='NETCDF4')

    strftime = []
    for i in range(len(Radar.time())):
        strftime.append(Radar.time()[i].strftime("%Y%m%d%H%M%S"))

    numtime = np.asarray(Radar.time("ut"), dtype="f8")
    strftime = np.asarray(strftime, dtype="S14")
    CM = np.asarray(Radar.cloudMask(),dtype="f8")
    RM = np.asarray(Radar.rainRate(),dtype="f8")
    range_values = np.asarray(Radar.range(),dtype="f4")
    BCOtime = np.asarray(Radar.BCOtime(),dtype="f8")

    # Create global attributes
    nc.location = "The Barbados Cloud Observatory, Deebles Point, Barbados"
    nc.converted_by = "Tobias Machnitzki (tobias.machnitzki@mpimet.mpg.de)"
    nc.institution = "Max Planck Institute for Meteorology, Hamburg"
    nc.created_with = os.path.basename(__file__) + " with its last modification on " + time.ctime(
        os.path.getmtime(os.path.realpath(__file__)))
    nc.creation_date = time.asctime()
    nc.version = "1.0.0"
    nc.derived_from = NC_FILE

    # Create dimensions
    time_dim = nc.createDimension('time', None)
    range_dim = nc.createDimension('range',None)

    # Create variable
    time_var = nc.createVariable('time', 'f8', ('time',))
    time_var.units = "Seconds since 1970-1-1 0:00:00 UTC"
    time_var.CoordinateAxisType = "Time"
    time_var.calendar = "Standard"
    time_var.Fill_value = "-999"

    range_var = nc.createVariable('range','f4', ('range',))
    range_var.units = "m"
    range_var.CoordinateAxisType = "Height"
    range_var.long_name = "Range from Antenna to the Centre of each Range Gate"
    range_var.Fill_value = "-999"

    strftime_var = nc.createVariable('strftime', 'S8', ('time',))
    strftime_var.units = "YYYYMMDDHHMMSS"
    strftime_var.CoordinateAxisType = "Time"

    BCOtime_var = nc.createVariable('bco_day','f8',('time',))
    BCOtime_var.long_name = "Days since start of Barbados Cloud Observatory measurements"
    BCOtime_var.units = "Days since 2010-4-1 00:00:00 (UTC)"

    CM_var = nc.createVariable('CloudMask','f4',('time','range'))
    CM_var.long_name = "Derived cloud mask"
    CM_var.description = "\n      0=Cloudbeard\n      1=Rain\n      2=Cirrus\n      3=Cumulus\n"

    RM_var = nc.createVariable('RainMask','f4',('time','range'))
    RM_var.long_name = "Derived rain rate from Marshall-Palmer relation"


    # Fill varaibles with values
    time_var[:] = numtime[:]
    strftime_var[:] = strftime[:]
    range_var[:] = Radar.range()[:]
    BCOtime_var[:] = Radar.BCOtime()[:]

    CM_var[:] = CM[:]
    RM_var[:] = RM[:]

    nc.close()

def smooth(Radar):
    CumulusCloudMask = Radar.cloudMask().copy()
    CumulusCloudMask[CumulusCloudMask != 3] = np.nan
    CumulusCloudMask[np.isnan(CumulusCloudMask)] = 0
    CumulusCloudMask =  binary_opening(CumulusCloudMask, iterations=2).astype(int)
    CumulusCloudMask = binary_closing(CumulusCloudMask, iterations=10).astype(float)
    CumulusCloudMask[CumulusCloudMask == 0] = np.nan

    CirrusCloudMask = Radar.cloudMask().copy()
    CirrusCloudMask[CirrusCloudMask != 2] = np.nan
    CirrusCloudMask[np.isnan(CirrusCloudMask)] = 0
    CirrusCloudMask =  binary_opening(CirrusCloudMask, iterations=5).astype(int)
    CirrusCloudMask = binary_closing(CirrusCloudMask, iterations=30).astype(float)
    CirrusCloudMask[CirrusCloudMask == 0] = np.nan

    CM_smooth = CirrusCloudMask.copy()
    CM_smooth[np.logical_or((CirrusCloudMask == 1),(CumulusCloudMask==1))] = 1
    return CM_smooth

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

    SAVE_PATH = ""
    SAVE_FILE = NC_FILE[:-3] + "_CloudMask.nc"

    # Initiate class:
    Radar = RadarClass(NC_PATH + NC_FILE)

    # create_netCDF(Radar,NC_FILE,SAVE_FILE,SAVE_PATH)

    # contourf_plot(Radar, Radar.rainRate())
    smoothed = smooth(Radar)
    smoothed[np.isnan(smoothed)] = 0
    mask = smoothed > np.mean(smoothed)
    structure_array = np.ones([3,3])
    label_im, nb_labels = ndimage.label(mask, structure=structure_array)
    print("Clouds in picture: %i" %nb_labels)
    label_im = label_im.astype(float)
    label_im[label_im == 0] = np.nan
    plt.contourf(label_im.transpose())

    # plotCloudmask(Radar, mask)
