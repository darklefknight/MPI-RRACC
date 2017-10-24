import os
from joblib import delayed,Parallel


def main(date,device):
    try:
        os.system("python3 RRACC.py {} --device {}".format(str(date),device))
    except:
        pass

if __name__ == "__main__":
    device = "MBR2"
    Parallel(n_jobs=1,verbose=5)(delayed(main)(date,device) for date in range(20150801,20150831,1) )