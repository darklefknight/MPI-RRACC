"""
Created by Tobias Machnitzki (tobias.machnitzki@mpimet.mpg.de) on the 27.10.2017

RRACC stands for "Radar Rain and Cloud Classification"

"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from datetime import datetime as dt
from calendar import timegm
import argparse
import sys
from joblib import Parallel,delayed

#----------------------------------------------------------------------------------------------------------------------
# =========================================
# Radar Class
# =========================================
class RadarClass:
    """
    Class for working with Radar data.
    For initialisation it needs the path of a netCDF-file with Radar-data.
    """
    def __init__(self,nc_FilePath):
        nc = Dataset(nc_FilePath)
        self._time = nc.variables['time'][:].copy() #time in sec since 1970
        self._range = nc.variables['range'][:].copy() #range in m
        self._Zf = nc.variables['Zf'][:].copy() #filtered reflectivity
        self._MeltHeight = nc.variables['MeltHei'][:].copy() # Height of the meltinglayer in m
        self._VEL = nc.variables['VEL'][:].copy() # "vertical velocity of all Hydrometeors"
        nc.close()

        self._cloudMask = np.asarray(self._Zf)

        self._obj_time = []
        self.__time2obj()

    def __time2obj(self):
        for element in self._time:
            self._obj_time.append(self.__dt_obj(element))
        self._obj_time = np.asarray(self._obj_time)

    def __dt_obj(self,u):
        return dt.utcfromtimestamp(u)

    def __ut_obj(self,d):
        return timegm(d.timetuple())

    def time(self,shape="dt"):
        """
        :param shape: what shape the returned values should have. Allowed:
                "dt" = datetime-object
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


# ----------------------------------------------------------------------------------------------------------------------

def contourf_plot(MBR2,value):
    plt.contourf(MBR2.time(),MBR2.range(),value.transpose(),cmap="jet")
    plt.ylim(0,10000)

def calculateCloudMask(i):
    for j in range(len(Radar.range())):
        if Radar.VEL()[i, j] < -1:
            cloudMask[i,j] =  0
        elif Radar.Zf()[i, j] > -50 and Radar.VEL()[i, j] > -1:
            cloudMask[i, j] = 30
        elif Radar.Zf()[i, j] < -50:
            cloudMask[i, j] = -50
        else:
            cloudMask[i,j] = np.nan


def getCloudMask(parallel=True):
    if parallel:
        Parallel(n_jobs=-1, verbose=5)(delayed(calculateCloudMask())(i) for i in range(len(Radar.time())))
    else:
        [calculateCloudMask(i) for i in range(len(Radar.time()))]

if __name__ == "__main__":

    # Get parsed arguments:
    parser = argparse.ArgumentParser(description="example: RRACC.py 20170401",prog="RRACC.py")
    parser.add_argument('datestr',metavar='t', help='add date-argument YYYYMMDD',type=int)
    parser.add_argument('--device', dest='device', help='specify a device. Allowed: MBR2, KATRIN. Default: MBR2', default="MBR2")
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

    #Initiate class:
    Radar = RadarClass(NC_PATH+NC_FILE)

    #TODO: Put this later in the RadarClass as function:
    cloudMask = np.asarray(Radar.Zf())
    getCloudMask(parallel=True)


    contourf_plot(Radar,cloudMask)
