from RRACC import RadarClass
import argparse
from datetime import timedelta
from datetime import datetime as dt
import numpy as np
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
import pandas as pd


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def getArrays(date):


    return (KATRIN_mean,MBR2_mean)

if __name__ == "__main__":

    # datestr = date.strftime("%y%m%d")
    datestr = "150801"

    NC_PATH1 = "/data/mpi/mpiaes/obs/m300517/RadarCorrection/MBR2_out/"
    # NC_FILE = "MMCR__MBR__Spectral_Moments__10s__155m-25km__" + datestr + ".nc"
    NC_FILE1 = "MMCR__MBR__Spectral_Moments__10s__155m-14km__" + datestr + ".nc"

    NC_PATH2 = "/data/mpi/mpiaes/obs/m300517/RadarCorrection/KATRIN_out/"
    NC_FILE2 = "MMCR__KATRIN__Spectral_Moments__10s__300m-15km__" + datestr + ".nc"

    try:
        MBR2 = RadarClass(NC_PATH1 + NC_FILE1)
        KATRIN = RadarClass(NC_PATH2 + NC_FILE2)

        KATRIN_mean = np.nanmean(KATRIN.Zf())
        MBR2_mean = np.nanmean(MBR2.Zf())
    except:
        KATRIN_mean = np.nan
        MBR2_mean = np.nan

    START_DATE = dt(2015,6,1)
    END_DATE = dt(2015,6,10)

    jobs = [x for x in daterange(START_DATE,END_DATE)]

    # means = Parallel(n_jobs=-1,)(delayed(getArrays)(date) for date in jobs)
    # meanArray = np.asarray(means,dtype=[("KATRIN","f8"),("MBR2","f8")])

    zfs = [x for x in range(-100,100,1)]

    MBR2_array = np.zeros([len(zfs),len(MBR2.range())],dtype="f8")
    KATRIN_array = np.zeros([len(zfs), len(KATRIN.range())],dtype="f8")

    df_K_Zf = pd.DataFrame(KATRIN.Zf().transpose())
    df_M_Zf = pd.DataFrame(MBR2.Zf().transpose())

    df_K = pd.DataFrame()
    df_M = pd.DataFrame()

    for i in range(-100,100,1):
        df_K[i] = df_K_Zf.isin([i]).sum(1)
        df_M[i] = df_M_Zf.isin([i]).sum(1)

