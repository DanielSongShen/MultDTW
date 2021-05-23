# this script collects the experiment results on one or more dataset and save the results to a separate place
# it takes in a list of numbers (e.g., 1,3,4) as the index ID of the datasets to process
# it uses one setting for one method.

import os
import pickle as pkl
import pandas as pd
import sys
import numpy as np
import glob
from Source import *

maxdim_g = 5
nqueries_g = 0
nreferences_g = 0
windows_g = [20]
machineRatios = [1, 1]
THs_g_TI = [0.1]
Ks_g = [6]
Qs_g = [2]
THs_g_PC = [0.5]
C_g = 0
period_g = 5

datapath="../Data/Multivariate_pickled/"
resultpath = "../Results/HurricaneCount/"
if (not os.path.exists(resultpath)):
    temp = False
    os.makedirs(resultpath)

datasetsNameFile = "../Data/useddatasets.txt"
with open (datasetsNameFile, 'w+') as f:
    f.write("HurricaneCount"+'\n')
f.close()

datasetsSizeFile = "../Data/useddatasets_size.txt"
with open (datasetsSizeFile, 'w+') as f:
    f.write(str(70)+'\n')
f.close()

nqnrfile = resultpath+'/usabledatasets_nq_nref.txt'

if nqueries_g*nreferences_g==0:
    with open(nqnrfile,'w+') as f:
        f.write(str(1)+'\n')
        f.write(str(30)+'\n')
    f.close()
#
"""
print(">>>>>> Start LB_MV_ws")
LB_MV_ws.dataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g)
LB_MV_ws.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)
#
print(">>>>>> Start LB_TI_ws")
LB_TI_ws.dataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g_TI)
LB_TI_ws.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g_TI)
#
print(">>>>>> Start LB_PC_ws")
LB_PC_ws.dataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_PC, C_g)
LB_PC_ws.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_PC, C_g)

print(">>>>>> Start LB_MV")
LB_MV.dataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g)
LB_MV.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g)
"""

"""
if (not os.path.exists(datapath+"HurrcaneCount")):
    os.makedirs(datapath+"HurricaneCount")
for i in range(1951, 2021):
    try:
        csvpath = "../Data/HurricaneCount/" + str(i) + ".csv"
        temp = pd.read_csv(csvpath, usecols=[1,2,3,4])
        filepath = datapath + "HurricaneCount/" + str(i)
        temp.to_pickle(filepath+".pkl")
        print()
#        print(temp)
    except:
        print("Error in loading/pickling data "+str(i))
"""

print(">>>>>> Start LB_TI")



# r = pd.read_csv(resultpath+"/d5/w20/allResults.csv")

# Util.initloadHurrianeKNNResults(resultpath+"120days/", 5)
Util.reverseDTW()
Util.HurricaneEvaluation()

# Data = first 120 days
# Top 5 nearest neighbors
# LB_TI.hurricaneDataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, days=120, N=5)
# Top 10 nearest neighbors
# LB_TI.hurricaneDataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, days=120, N=10)

# Data = first 365 days
# Top 5 nearest neighbors
# LB_TI.hurricaneDataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, days=365, N=5)
# Top 10 nearest neighbors
# LB_TI.hurricaneDataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, days=365, N=10)

"""
LB_TI.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g_TI)

print(">>>>>> Start LB_PC")
LB_PC.dataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_PC)
LB_PC.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_PC)

dir = '/d'+str(maxdim_g)+'/w'+ str(windows_g[0]) + '/'
# validate the consistency of the results of all methods that use pre-setups
if sum(Util.checkRst(resultpath, dir, ["HurricaneCount"], str(nqueries_g) + 'X' + str(nreferences_g)+'*_ws_*results.txt')):
    print ('WS methods have disparities!!!')
else:
    print ('WS methods have consistent results.')
# validate the consistency of the results of all methods that don't use pre-setups
if sum(Util.checkRst_n (resultpath, dir, ["HurricaneCount"], '!('+str(nqueries_g) + 'X' + str(nreferences_g)+'*_ws_*results.txt)')):
    print ('Online methods have disparities!!!')
else:
    print ('Online methods have consistent results.')
"""
print("End")