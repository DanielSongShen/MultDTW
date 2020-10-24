import glob

import numpy as np

path = "../Results/_AllDataSets/d5/"
settingfiles = glob.glob(path+'*_settingchosen.npy')
for f in settingfiles:
    setting = np.load(f)
    print(f+':')
    print(setting)