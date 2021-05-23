# this script collects the experiment results on one or more dataset and save the results to a separate place
# it takes in a list of numbers (e.g., 1,3,4) as the index ID of the datasets to process
# it uses one setting for one method.

import os
import sys
import numpy as np
# from Source import *


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


argvs = sys.argv
args = len(argvs)
if (args<2):
    print('Please indicate the ID numbers of the datasets (e.g., 1,3) to work on.')
    exit()
datasetsIDs = argvs[1].split(',')
datasetsIDs = [int(x) for x in datasetsIDs]

datapath="../Data/Multivariate_pickled/"
alldatasetsNameFile = "../Data/datasets.txt"
alldatasetsSizeFile = "../Data/datasets_size.txt"
resultpath = "../Results/tryonesetting/"
if (not os.path.exists(resultpath)):
    os.makedirs(resultpath)

datasetsNameFile = "../Data/useddatasets.txt"
datasetsSizeFile = "../Data/useddatasets_size.txt"
nqnrfile = resultpath+'/usabledatasets_nq_nref.txt'

with open(alldatasetsNameFile, 'r') as f:
    alldatasets = f.read().strip()
alldatasets = alldatasets.split('\n')
thisrunsets = [alldatasets[i-1] for i in datasetsIDs]
np.savetxt(datasetsNameFile, thisrunsets, fmt='%s')

with open(alldatasetsSizeFile, 'r') as f:
    alldatasizes = f.read().strip()
alldatasizes=alldatasizes.split('\n')
thisrunsizes = [alldatasizes[i-1] for i in datasetsIDs]
np.savetxt(datasetsSizeFile, thisrunsizes, fmt='%s')

if nqueries_g*nreferences_g==0:
    with open(nqnrfile,'w+') as f:
        for s in thisrunsizes:
            s = int(s)
            f.write(str(int(s*0.3))+'\n')
            f.write(str(s-int(s*0.3))+'\n')

print(">>>>>> Measure 1 DTW times.")
Measure1DTWtime.MeasurePrimeTimes \
    (resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, windows_g[0], Ks_g, Qs_g)

#
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

print(">>>>>> Start LB_TI")
LB_TI.dataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g_TI)
LB_TI.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g_TI)

print(">>>>>> Start LB_PC")
LB_PC.dataCollection(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_PC)
LB_PC.dataProcessing(datasetsNameFile, resultpath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_PC)

dir = '/d'+str(maxdim_g)+'/w'+ str(windows_g[0]) + '/'
# validate the consistency of the results of all methods that use pre-setups
if sum(Util.checkRst(resultpath, dir, thisrunsets, str(nqueries_g) + 'X' + str(nreferences_g)+'*_ws_*results.txt')):
    print ('WS methods have disparities!!!')
else:
    print ('WS methods have consistent results.')
# validate the consistency of the results of all methods that don't use pre-setups
if sum(Util.checkRst_n (resultpath, dir, thisrunsets, '!('+str(nqueries_g) + 'X' + str(nreferences_g)+'*_ws_*results.txt)')):
    print ('Online methods have disparities!!!')
else:
    print ('Online methods have consistent results.')

print("End")