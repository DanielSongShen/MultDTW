from Source.Util import *

# This file implements LB_TI with early abandoning


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


def DTWwnd (s1, s2, windowSize):
    '''
    Compute the DTW distance between s1 and s2, and also the neighbor distances of s1
    :param s1: a series
    :param s2: a series
    :param windowSize: half window size
    :return: DTW distance, neighbor distances
    '''
    DTW = {}
    dxx = []
    w = max(windowSize, abs(len(s1)-len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)-1):
        dxx.append(distance(s1[i+1], s1[i]))
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    # final iteration
    i = len(s1)-1
    for j in range(max(0, i - w), min(len(s2), i + w)):
        dist = distance(s1[i], s2[j])
        DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return DTW[len(s1)-1, len(s2)-1], dxx


def tiBounds_top_calP_list_comp_eb(X, Y, W, P, dxx, dist):
    # Same as tiBounds except that the true distances are calculated in every P samples of X
    # And early abondoning is used.
    Xlen = list(X.shape)[0]
    Ylen = list(Y.shape)[0]

    upperBounds = np.zeros([Xlen, W*2+1])
    lowerBounds = np.zeros([Xlen, W*2+1])
#    overallLowerBounds = np.zeros(Xlen)

    lbrst = 0
    for t in range(0, Xlen):
        startIdx = 0 if t > W else W - t
        if t % P == 0:
            lw = max(0, t - W)
            tp = min(t + W + 1, Ylen)
            dxyInit = np.array([distance(X[t, :], Y[i, :]) for i in range(lw, tp)])

            upperBounds[t, startIdx:startIdx + tp - lw] = dxyInit
            lowerBounds[t, startIdx:startIdx + tp - lw] = dxyInit
            lbrst+= np.amin(dxyInit)
        else:
            startIdx = 0 if t > W else W - t
            lr = 0 if t < W else t - W
            ur = Ylen - 1 if Ylen - 1 < t + W else t + W
            thisdxx = dxx[t - 1]
            startIdx_lr = startIdx - lr + 1
            t_1 = t - 1
            idx = ur - lr - 1
            if t + W <= Ylen - 1:
                upperBounds[t, startIdx:startIdx + ur - lr] = [upperBounds[t_1, startIdx_lr + i] + thisdxx for i in
                                                               range(lr, ur)]
                lowerBounds[t, startIdx:startIdx + ur - lr] = \
                    [lowerBounds[t_1, startIdx_lr + i] - thisdxx if lowerBounds[t_1, startIdx_lr + i] > thisdxx
                     else 0 if thisdxx < upperBounds[t_1, startIdx_lr + i] else thisdxx - upperBounds[
                        t_1, startIdx_lr + i]
                     for i in range(lr, ur)]
                # the last y point
                temp = distance(X[t, :], Y[ur, :])
                upperBounds[t, startIdx + idx + 1] = temp
                lowerBounds[t, startIdx + idx + 1] = temp
                lbrst+= np.amin(lowerBounds[t, startIdx:startIdx + idx + 2])
            else:
                upperBounds[t, startIdx:startIdx + idx + 2] = [upperBounds[t_1, startIdx_lr + i] + thisdxx for i in
                                                               range(lr, ur + 1)]
                lowerBounds[t, startIdx:startIdx + idx + 2] = \
                    [lowerBounds[t_1, startIdx_lr + i] - thisdxx if lowerBounds[t_1, startIdx_lr + i] > thisdxx
                     else 0 if thisdxx < upperBounds[t_1, startIdx_lr + i] else thisdxx - upperBounds[
                        t_1, startIdx_lr + i]
                     for i in range(lr, ur + 1)]
                lbrst+= np.amin(lowerBounds[t, startIdx:startIdx + idx + 2])
        if lbrst>=dist:
            return lbrst   # early abandoning
    #------------
    return lbrst

def DTWDistanceWindowLB_Ordered_LBTI_ (queryID, TH, P, query, references, W):
    '''
    Compute the DTW distance between a query series and a set of reference series.
    :param i: the query ID number
    :param DTWdist: precomputed DTW distances (for fast experiments)
    :param TH: the triggering threshold for the expensive filter to take off
    :param query: the query series
    :param references: a list of reference series
    :param W: half window size
    :return: the DTW distance and the coretime
    '''
    skips = 0
    p_cals = 0
    coretime = 0

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
#    end=time.time()
#    coretime += (end - start)

    dist,dxx  = DTWwnd (query, references[predId], W)

#    start = time.time()
    for x in range(1, len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if LBs[thisrefid] >= dist:
            skips = len(LBs) - x
            break
        elif LBs[thisrefid] >= dist - TH*dist:
            p_lb = tiBounds_top_calP_list_comp_eb(query, references[thisrefid], P, W, dxx, dist)
            p_cals += 1
            if p_lb < dist:
                dist2 = DTW_a(query, references[thisrefid], W, dist)
                if dist > dist2:
                    dist = dist2
                    predId = thisrefid
            else:
                skips = len(LBs) - x
                break
        else:
            dist2 = DTW_a(query, references[thisrefid], W, dist)
            if dist > dist2:
                dist = dist2
                predId = thisrefid

    end = time.time()
    coretime += (end - start)
#    LBs_g.append(LBs)

    return dist, predId, skips, coretime, p_cals

def dataCollection (pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], THs=[0.1], period_g=5):
    datasets = []
    # with open("Results/UCR/allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize = []
    # with open("Results/UCR/size.txt",'r') as f:
    with open(datasetsSizeFile, 'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

#    datasets=["ArticularyWordRecognition","AtrialFibrillation"]


    for idxset, dataset in enumerate(datasets):
        print(dataset+" Start!")
        assert(datasize[idxset]>=nqueries+nreferences)
        stuff = loadUCRData_norm_xs(datapath, dataset, nqueries+nreferences)
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
        # -------------------------------------------------

        print(dataset+":  "+ str(nqueries)+" queries, "+ str(nreferences)+ " references." +
              " Total dtw: "+str(nqueries*nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w)+"/"
#            dists = [[DTW(s1, s2, windowSize) for s2 in reference] for s1 in query]
            for TH in THs:
                results = [DTWDistanceWindowLB_Ordered_LBTI_ (ids1, TH,
                            period_g, query[ids1], reference, windowSize) for ids1 in range(len(query))]
                # if findErrors(dataset, maxdim, w, nqueries, nreferences, results, pathUCRResult):
                #     print('Wrong Results!! Dataset: ' + dataset)
                #     exit()
                with open(toppath+ str(nqueries) + "X" + str(
                    nreferences) + "_LBTI_TH"+str(TH)+"_results.txt", 'w') as f:
                    for r in results:
                        f.write(str(r)+'\n')
                f.close()
                # with open(toppath+ str(nqueries) + "X" + str(
                #     nreferences) + "_X1TH"+str(TH)+"_times.txt", 'w') as f:
                #     f.write(str(end-start)+'\n')
                #     f.write(str(periodTime)+'\n')
                # f.close()
                # allResults.append(results)
#    np.save(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" + str(nqueries)+"X"+str(nreferences)
#            + "_X1"+"w" + intlist2str(windows)+ "TH"+intlist2str(THs) + "_times.npy", allTimes)
    return 0


def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], THs=[0.1], machineRatios=[1,1]):
    datasets = []
    # with open(pathUCRResult+"allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    window = windows[0]
    rdtw = machineRatios[0]
    rother = machineRatios[1]
    t1dtw = loadt1dtw(pathUCRResult, maxdim, window)

#    datasets = ["ArticularyWordRecognition", "AtrialFibrillation"]

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
        for TH in THs:
            results = readResultFile(
            pathUCRResult + dataset + '/d' + str(maxdim) + "/w" + str(windows[0]) + "/" + str(nqueries) + "X" + str(
                nreferences) + "_LBTI_TH" + str(TH) + "_results.txt")
            tCore.append(sum(results[:, 3]))
            skips.append(sum(results[:, 2]))
    tCore = np.array(tCore).reshape((ndatasets,-1))
    skips = np.array(skips).reshape((ndatasets,-1))

    tCorePlus = tCore
#    tDTW = np.tile(t1dtw,(skips.shape[1],1)).transpose() * ((skips-totalPairs)*-1)
    tsum = rother*tCorePlus
    tsum_min = np.min(tsum,axis=1)
    setting_chosen = np.argmin(tsum, axis=1)
    skips_chosen = np.array( [skips[i,setting_chosen[i]] for i in range(skips.shape[0])] )
#    overhead = rother* np.array([tCorePlus[i,setting_chosen[i]] for i in range(tCorePlus.shape[0])])
    speedups = (rdtw*t1dtw[0:ndatasets] * NPairs) / tsum_min
#    overheadrate = overhead/(rdtw*t1dtw * NPairs)

    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBTI_w" + str(window) + 'TH'+intlist2str(THs)+'_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBTI_w" + str(window) + 'TH' + intlist2str(THs) + '_skipschosen.npy', skips_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBTI_w" + str(window) + 'TH' + intlist2str(THs) + '_settingchosen.npy', setting_chosen)
#    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
#            "_LBTI_w" + str(window) + 'TH' + intlist2str(THs) + '_overheadrate.npy', overheadrate)
    return 0
