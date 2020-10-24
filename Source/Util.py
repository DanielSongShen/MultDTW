import math
import os
import numpy as np
import pandas as pd
import glob

import time

import re


def calNeighborDistances(A):
    aa = [distance(A[i,:], A[i+1,:]) for i in range(0, len(A)-1)]
    return aa

def distance(p1, p2):
    x = 0
    for i in range(len(p1)):
        x += (p1[i] - p2[i]) ** 2
    return math.sqrt(x)

def DTW(s1, s2, windowSize):
    DTW = {}
    w = max(windowSize, abs(len(s1)-len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return DTW[len(s1)-1, len(s2)-1]


def DTW_a(s1, s2, windowSize, bestdist):
    # DTW with early abandoning
    DTW = {}
    w = max(windowSize, abs(len(s1)-len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        d=float('inf')
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            if (d>DTW[(i,j)]):
                d = DTW[(i,j)]
        if d>=bestdist:
            return d
    return DTW[len(s1)-1, len(s2)-1]

def DTWDistanceWindowLB_Ordered_xs(queryID, LBs, DTWdist):
    skips = 0

    start = time.time()
    LBSortedIndex = np.argsort(LBs)
    #LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
#    dist = DTW(query, references[predId], w)
    dist = DTWdist[queryID][predId]   # xs: changed
    for x in range(1,len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if dist>LBs[thisrefid]:
#           Use saved DTW distances from baseline
            dist2 = DTWdist[queryID][thisrefid]
            if dist>dist2:
                dist = dist2
                predId = thisrefid
        else:
            skips = len(LBSortedIndex) - x
            break
    end = time.time()
    coreTime = end - start
    return dist, predId, skips, coreTime

def DTWDistanceWindowLB_Ordered_xs_a(LBs, w, query, references):
    skips = 0

    start = time.time()
    LBSortedIndex = np.argsort(LBs)
    #LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
    dist = DTW(query, references[predId], w)
    for x in range(1,len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if dist>LBs[thisrefid]:
            dist2 = DTW_a(query, references[thisrefid],w, dist)
            if dist>dist2:
                dist = dist2
                predId = thisrefid
        else:
            skips = len(LBs) - x
            break
    end = time.time()
    coreTime = end - start
    return dist, predId, skips, coreTime

def get_skips (dataset, maxdim, w, lbs, queries, references, pathUCRResult, nq, nref):
    nqueries=len(queries)
    nrefs=len(references)
    print("W="+str(w)+'\n')
    distanceFileName = pathUCRResult + dataset + '/d' + str(maxdim) + '/w'+ str(w) + "/"+str(nq)\
                       +"X"+str(nref)+"_NoLB_DTWdistances.npy"
    print(distanceFileName)
    assert(os.path.exists(distanceFileName))
    # if not os.path.exists(distanceFileName):
    #     print('get_skips: found no distance file. recollect distances.')
    #     distances = [[DTW(s1, s2, w) for s2 in references] for s1 in queries]
    #     np.save(distanceFileName,np.array(distances))
    # else:
    print('get_skips: loading distances.')
    distances = np.load(distanceFileName)

    results =[]
    for ids1 in range(nqueries):
        results.append(DTWDistanceWindowLB_Ordered_xs(ids1, lbs[ids1], distances))
    return results

def get_skips_a(w, lbs, queries, references):
    nqueries=len(queries)

    results =[]
    for ids1 in range(nqueries):
        if ids1==5812:
            print('581th query done.')
            exit()
        rst = DTWDistanceWindowLB_Ordered_xs_a(lbs[ids1], w, queries[ids1], references)
        results.append(rst)
    return results


def loadt1dtw(pathUCRResult, maxdim, window):
    '''
    Load the time of one DTW for all datasets
    :param maxdim:
    :param window:
    :return: an nd array with all the times included
    '''
    t1dtwFile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+ '/Any_Anyw'+str(window)+'_t1dtw.npy'
    t1dtw = np.load(t1dtwFile)
    return t1dtw

def loadt1nd (pathUCRResult, maxdim, window):
    '''
    Load the time of one neighbor distance for all datasets.
    :param maxdim:
    :param window:
    :return: an nd array with all the times included
    '''
    t1ndFile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+ '/Any_Anyw'+str(window)+'_t1nd.npy'
    t1nd = np.load(t1ndFile)
    return t1nd

def DTWwlb(s1,s2,hwindowSize):
    '''
    Compute DTW between q and r and also the tight lower bound between them
    :param s1: query series
    :param s2: reference series
    :param hwindowSize: half window size
    :return: dtw distance, tight lower bound
    '''
    DTW = {}
    w = max(hwindowSize, abs(len(s1)-len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')

    DTW[(-1, -1)] = 0

    lb = 0
    for i in range(len(s1)):
        mn=float("inf")
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            mn=dist if dist<mn else mn
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
        lb+=mn
    return DTW[len(s1)-1, len(s2)-1], lb

def edist(A,B):
    '''
    Compute the enclean distance between two np arrays
    :param A: an np array
    :param B: an np array
    :return: their enclean distance
    '''
    return sum([math.sqrt(sum( [(A[i][j]-B[i][j])**2 for j in range(A.shape[1])] )) for i in range(A.shape[0])])

def normalize(aserie):
    # data structure: DataFrame [ dim1 [...], dim2 [...], ...] ]
    nmSerie = []
    for d in range(aserie.shape[1]):
        oneDim = list(aserie[d])
        mi = min(oneDim)
        ma = max(oneDim)
        dif = (ma - mi) if (ma-mi)>0 else 0.0000000001
        nmValue = [(x-mi)/dif for x in oneDim]
        nmSerie.append(nmValue)
    return pd.DataFrame(nmSerie).transpose()

def loadUCRData_norm_xs (path, name, n):
    # dataDims = pd.read_csv(path + "DataDimensions.csv")
    # dataDims = dataDims.drop(columns=dataDims.columns[10:])
    # dataDims.at[23, "Problem"] = "PhonemeSpectra"
    # dataDims = dataDims.set_index('Problem')
    # dataDims['Total Instances'] = [(datafile[1] + datafile[2]) * datafile[3] * datafile[4] for id, datafile in dataDims.iterrows()]

    datasetName = name
    if n==0:
        pklfiles = glob.glob(path + datasetName + "/*.pkl")
        pklfiles = sorted(pklfiles, key=lambda x: float(re.findall("(\d+)", x)[-1]))
        allData = [normalize(pd.read_pickle(g).fillna(0)) for g in pklfiles]
    else:
        cnt = 0
        allData = []
        pklfiles = glob.glob(path + datasetName + "/*.pkl")
        pklfiles = sorted(pklfiles, key=lambda x: float(re.findall("(\d+)", x)[-1]))
        for g in pklfiles:
            cnt +=1
            if (cnt > n):
                break
            else:
                allData.append(normalize(pd.read_pickle(g).fillna(0)))
    return allData

def intlist2str(A):
    return '_'.join([str(a) for a in A])

def load_M0LBs(pathUCRResult, dataset, maxdim, w, nqueries, nreferences):
    lb_2003 = np.load(pathUCRResult+dataset+"/d"+ str(maxdim) +"/w"+ str(w) + '/' +
                      str(nqueries) + "X" + str(nreferences) +"_LBMV_ws_lbs.npy")
    return lb_2003

def getGroundTruth (dataset, maxdim, w, nqueries, nreferences, pathUCRResult='../Results/UCR/'):
    '''
    Assuming that the DTW distances between all queries and all references are already available. This function
    generates the ground truth of the nearest distance and neighbor for each query. It outputs the results to
    a text file, and an npy file.
    :param dataset:
    :param maxdim:
    :param w:
    :param nqueries:
    :param nreferences:
    :return: 0
    '''
    distanceFileName = pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                       str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
    textoutputFile = pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                       str(nqueries) + "X" + str(nreferences) + "_NoLB_results.txt"
    distances = np.load(distanceFileName)
    minDistances = np.min(distances, axis=1).tolist()
    indices = np.argmin(distances, axis=1).tolist()
    results = zip(minDistances, indices)
    with open(textoutputFile,'w') as f:
        for r in results:
            f.write(str(r) + '\n')
#    np.save(npyoutputFile, np.array(results))


def checkRst_n(resultpath, dir, datasets, pattern):
    '''
    Check the consistency of the DTW results across methods
    :param prefix: the path containing the results
    :param datasets: the datasets to examine
    :param pattern: the pattern of file to exclude
    :return: True or False
    '''
    hasErrors = []
    for ds in datasets:
        haserr = False
        rstfiles = list(set(glob.glob(resultpath + ds + "/" + dir + '*_results.txt')) - set(glob.glob(resultpath + ds + "/" + dir + pattern)))
        results0 = readResultFile(rstfiles[0])
        for otherrstfile in rstfiles[1:]:
            resultsOther = readResultFile(otherrstfile)
            for i in range(results0.shape[0]):
                if (results0[i, 0] != resultsOther[i, 0]):
                    print(ds+': Disparity found on query '+ str(i) + ' between ' + rstfiles[0] + ' and ' + otherrstfile)
                    haserr = True
        hasErrors.append(haserr)
    return hasErrors


def checkRst(resultpath, dir, datasets, pattern):
    '''
    Check the consistency of the DTW results across methods
    :param prefix: the path containing the results
    :param datasets: the datasets to examine
    :param pattern: the pattern of file to check
    :return: True or False
    '''
    hasErrors = []
    for ds in datasets:
        haserr = False
        rstfiles = glob.glob(resultpath + ds + "/" + dir + pattern)
        results0 = readResultFile(rstfiles[0])
        for otherrstfile in rstfiles[1:]:
            resultsOther = readResultFile(otherrstfile)
            for i in range(results0.shape[0]):
                if (results0[i, 0] != resultsOther[i, 0]):
                    print(ds+': Disparity found on query '+ str(i) + ' between ' + rstfiles[0] + ' and ' + otherrstfile)
                    haserr = True
        hasErrors.append(haserr)
    return hasErrors


def getGroundTruth_allDataSets (maxdim, windows, nqueries, nreferences, pathUCRResult='../Results/UCR/'):
    datasets = []
    with open(pathUCRResult + "allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    for idxset, dataset in enumerate(datasets):
        print(dataset + " Start!")
        for w in windows:
            getGroundTruth (dataset, maxdim, w, nqueries, nreferences, pathUCRResult)

def readResultFile (f):
    '''
    Read in a result file and store it into an nd array
    :param f: the result file name
    :return: an nd array
    '''
    list = []
    with open(f,'r') as f:
        lines = f.readlines()
        for ln in lines:
            ln = ln.strip().strip("(").strip(")")
            if ln!='':
                list.append([float(a) for a in ln.split(',')])
    return (np.array(list))

