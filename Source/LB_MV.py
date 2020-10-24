from Source.Util import *

# This file implements quantization-based clustering for LB_MV.
# It is the online version, with adaptive cluster numbers.

# The dataCollection function saves the following:
#     the DTW distances, nearest neighbors, skips and coreTime in each individual directory: a text file

def getLB_oneQ_qbox (X, others, qbounds):
    '''
    Get the lower bounds between one query series X and many candidate series in others
    :param X: one series
    :param others: all candidate series
    :param qbounds: the bounding boxes of the query windows
    :return: the lower bounds between X and each candidate series
    '''
    lbs = []
    dim = len(X[0])
    for idy, s2 in enumerate(others):
        LB_sum = 0
        for idy, y in enumerate(s2):
            l=qbounds[idy][0]
            u=qbounds[idy][1]
            temp = math.sqrt(
                sum([(y[idd] - u[idd]) ** 2 if (y[idd] > u[idd]) else (l[idd] - y[idd]) ** 2 if (y[idd] < l[idd]) else 0
                     for idd in range(dim)]))
            LB_sum += temp
        lbs.append(LB_sum)
    return lbs

def DTWDistanceWindowLB_Ordered_LBMV (i, query, references, W):
    '''
    Compute the DTW distance between a query series and a set of reference series.
    :param i: the query ID number
    :param DTWdist: precomputed DTW distances (for fast experiments)
    :param query: the query series
    :param references: a list of reference series
    :param W: half window size
    :return: the DTW distance and the coretime
    '''
    skip = 0

    start = time.time()
    # get bounds of query
    ql = len(query)
    dim = len(query[0])
    bounds = []
    for idx in range(ql):
        segment = query[(idx - W if idx - W >= 0 else 0):(idx + W + 1 if idx + W <= ql-1 else ql)]
        l = [min(segment[:, idd]) for idd in range(dim)]
        u = [max(segment[:, idd]) for idd in range(dim)]
        bounds.append([l, u])
    LBs = getLB_oneQ_qbox(query, references, bounds)
    LBSortedIndex = np.argsort(LBs)
#    LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
    dist = DTW (query, references[predId], W)
    for x in range(1,len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if dist>LBs[thisrefid]:
#           Use saved DTW distances from baseline for quick experiment
#            dist2 = DTWdist[i][thisrefid]
            dist2 = DTW_a(query,references[thisrefid],W,dist)
            if dist>dist2:
                dist = dist2
                predId = thisrefid
        else:
            skip = len(LBs) - x
            break
    end = time.time()
    coreTime = end - start
#    LBs_g.append(LBs)

    return dist, predId, skip, coreTime


def dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20]):
    datasets=[]
    #with open("Results/UCR/allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    #with open("Results/UCR/size.txt",'r') as f:
    with open(datasetsSizeFile,'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

#    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

    # # create directories if necessary
    # for datasetName in datasets:
    #     for w in windows:
    #         dir = pathUCRResult+"" + datasetName + "/" + str(w)
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)

    for idxset, dataset in enumerate(datasets):
        print(dataset+" Start!")
        assert(datasize[idxset]>=nqueries+nreferences)
        stuff = loadUCRData_norm_xs(datapath, dataset,nqueries+nreferences)
        size = len(stuff)
        length = stuff[0].shape[0]
        dim = min(stuff[0].shape[1], maxdim)
        print("Size: "+str(size))
        print("Dim: "+str(dim))
        print("Length: "+str(length))
        samplequery = stuff[:nqueries]
        samplereference = stuff[nqueries:nreferences+nqueries]
        # -------------------------------------------------
        if (nqueries * nreferences == 0):  # all series to be used
            qfrac = 0.3
            samplequery = stuff[:int(size * qfrac)]
            samplereference = stuff[int(size * qfrac):]
            with open(pathUCRResult + '/usabledatasets_nq_nref.txt', 'a') as f:
                f.write(str(int(size * qfrac)) + "\n")
                f.write(str(size - int(size * qfrac)) + "\n")
        ##-------

        # -------------------------------------------------

        print(dataset+":  "+ str(nqueries)+" queries, "+ str(nreferences)+ " references." +
              " Total dtw: "+str(nqueries*nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w) + "/"
            # distanceFileName = pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
            #                    str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
            # assert(os.path.exists(distanceFileName))
            # distances = np.load(distanceFileName)

            results = [DTWDistanceWindowLB_Ordered_LBMV (ids1,
                                                      query[ids1], reference, windowSize) for ids1 in range(len(query))]
            # if findErrors(dataset,maxdim,w,nqueries,nreferences,results,pathUCRResult):
            #     print('Wrong Results!! Dataset: '+dataset)
            #     exit()
#            np.save(pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
#                    + str(nqueries) + "X" + str(nreferences) + "_LBMV_a_lbs.npy", np.array(LBs_g))
            with open(toppath + str(nqueries) + "X" + str(
                    nreferences) + "_LBMV_a" + "_results.txt", 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')
            f.close()

        print(dataset+" Done!"+'\n'+'\n')

    return 0



def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], machineRatios=[1,1]):
    '''
    Process the data to get the speedups. Currently, only deals with the first element in windows.
    :param datasetsNameFile:
    :param pathUCRResult:
    :param maxdim:
    :param nqueries:
    :param nreferences:
    :param windows:
    :param machineRatios: Used for cross-machine performance estimation. [r1, r2].
                          r1: tDTW(new machine)/tDTW(this machine);
                          r2: tM0LB(new machine)/tM0LB(this machine), taken as the ratio for all other times.
    :return: 0
    '''
    datasets=[]
    #with open(pathUCRResult+"allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    window = windows[0]
    rdtw = machineRatios[0]
    rother = machineRatios[1]
    t1dtw = loadt1dtw(pathUCRResult, maxdim, window)

#    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

    ndatasets = len(datasets)

    # compute speedups
    tCore = []
    skips = []

    ## -------------------
    NPairs = []
    if nqueries * nreferences == 0:
        actualNQNRs = np.loadtxt(pathUCRResult + '/usabledatasets_nq_nref.txt').reshape((-1, 2))
        for i in range(len(datasets)):
            actualNQ = actualNQNRs[i][0]
            actualNR = actualNQNRs[i][1]
            NPairs.append(actualNQ * actualNR)
    ## -------------------
    for dataset in datasets:
        results = readResultFile(pathUCRResult + dataset + '/d' + str(maxdim) + "/w"+ str(windows[0]) + "/" + str(nqueries) + "X" + str(nreferences)
            + "_LBMV_a" + "_results.txt")
        tCore.append(sum(results[:,3]))
        skips.append(sum(results[:,2]))
    tCore = np.array(tCore)
#    tDTW = t1dtw*(NPairs - np.array(skips))
    speedups = (rdtw*t1dtw[0:ndatasets]*NPairs)/(rother*tCore)
#    overheadrate = (rother*tCore)/(rdtw*t1dtw*NPairs)

    np.save(pathUCRResult+"_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBMV_a_w"+str(window)+'_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBMV_a_w" + str(window) + '_skips.npy', skips)
#    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
#            "_LBMV_a_w" + str(window) + '_overheadrate.npy', overheadrate)
    return 0

