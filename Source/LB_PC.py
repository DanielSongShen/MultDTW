from Source.Util import *
import time

# This file implements Z0 followed by quantization-based clustering for cases that LBMV fails but not miserably.
# The dataCollection function saves the following:
#     the DTW distances, nearest neighbors, skips, coreTime, Ze3 invoked times in each individual directory: a text file


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

def getLB_oneQR (Q, R, bboxes, dist):
    '''
    Get the lower bounds between two series, Q and R with multiple bounding boxes.
    :param Q: A series.
    :param R: A series.
    :param bboxes: the bounding boxes of Q.
    :return: the lower bound between Q and R
    '''
    #  X and Y one series, is all references, dim is dimensions, sl_bounds has all the bounding boxes of all reference series
    LB_sum = 0
    dim = len(Q[0])

    for idr, r in enumerate(R):
        numBoxes = len(bboxes[idr])
        bounds=[]
        for idbox in range(numBoxes):
            l = bboxes[idr][idbox][0]
            u = bboxes[idr][idbox][1]
            temp = math.sqrt(sum([(r[idd]-u[idd]) ** 2 if (r[idd] > u[idd]) else (l[idd]-r[idd])**2
                                    if (r[idd] < l[idd]) else 0 for idd in range(dim)]))
            bounds.append(temp)

        LB_sum+=min(bounds)
        if LB_sum >= dist:
            return LB_sum     # early abandoning
    return LB_sum

def inBoxes (p, bboxes):
    '''
    Check whether point p falls into boxes
    :param p:
    :param boxes: [ [several(dim) lowends] [several(dim) highends] ... ]
    :return: True or False
    '''
    dim = len(p)
    for box in bboxes:
        rst = [p[idd] > box[1][idd] or p[idd] < box[0][idd] for idd in range(dim)]
        s = sum(rst)
        if (s<1):
            return True
    return False

def findboxes(awindow, K, Q):
    '''
    Find the bounding boxes in a segment of a series
    :param: awindow: a segment of a series
    :param: K: the maximum number of boxes to find
    :param: Q: the maximum number of quantization levels per dimension
    :return: a list of bounding boxes
    '''
    cellMembers = {}
    bboxes = []
    dims = len(awindow[0])
    overall_ls = [min(np.array(awindow)[:, idd]) for idd in range(dims)]
    overall_us = [max(np.array(awindow)[:, idd]) for idd in range(dims)]
    cells = [1 + int((overall_us[idd] - overall_ls[idd]) * Q) for idd in range(dims)] # adaptive no. of cells
    celllens = [(overall_us[idd] - overall_ls[idd]) / cells[idd] + 0.00000001 for idd in range(dims)]
    for e in awindow:
        thiscell = str([int((e[idd] - overall_ls[idd]) / celllens[idd]) for idd in range(dims)])
        if thiscell in cellMembers:
            cellMembers[thiscell].append(e)
        else:
            cellMembers[thiscell] = [e]
    for g in cellMembers:
        l = [min(np.array(cellMembers[g])[:, idd]) for idd in range(dims)]
        u = [max(np.array(cellMembers[g])[:, idd]) for idd in range(dims)]
        bboxes.append([l, u])
    if len(bboxes) > K:
        # combine all boxes except the first K-1 boxes
        sublist = bboxes[K - 1:]
        combinedL = [min([b[0][idd] for b in sublist]) for idd in range(dims)]
        combinedU = [max([b[1][idd] for b in sublist]) for idd in range(dims)]
        bboxes = bboxes[0:K - 1]
        bboxes.append([combinedL, combinedU])
    return bboxes

def DTWwbbox (s1,s2,windowSize, K, Q):
    '''
    Compute DTW between q and r and also the tight lower bound between them
    :param s1: query series
    :param s2: reference series
    :param windowSize: half window size
    :return: dtw distance, bounding boxes
    '''
    DTW = {}
    bboxes = []
    w = max(windowSize, abs(len(s1)-len(s2)))
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
        left = max(0,i-w)
        right = min(len(s2),i+w)
        for j in range(left,right):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
        bboxes.append(findboxes(s1[left:right],K,Q))

    return DTW[len(s1)-1, len(s2)-1], bboxes

def DTWDistanceWindowLB_Ordered_LBPC_ (queryID, K, Q, query, references, W, TH=1):
    '''
    Compute the shortest DTW between a query and references series.
    :param queryID: the index number of this query
    :param DTWdist: the DTW distances between this query and each reference series (to avoid recomputing
                    the distance in each experiment)
    :param K: the maximum number of clusters
    :param Q: the quantization level
    :param s1: the query
    :param refs: the references series
    :param W: the half window size
    :param TH: the triggering threshold of more expensive lower bound calculations
    :return: the DTW distance, the neareast neighbor of this query, the number of DTW distance calculations skipped,
             the number of times the clustering-based method is invoked
    '''
    skip = 0
    cluster_cals = 0
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
    M0LBs = getLB_oneQ_qbox(query, references, bounds)
    LBSortedIndex = np.argsort(M0LBs)
#    LBSortedIndex = sorted(range(len(M0LBs)),key=lambda x: M0LBs[x])

    predId = LBSortedIndex[0]
#    end = time.time()
#    coretime += (end - start)

    dist, bboxes = DTWwbbox (query, references[predId], W, K, Q)

#    start = time.time()
    for x in range(1, len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if M0LBs[thisrefid] >= dist:
            skip += (len(M0LBs) - x)
            break
        elif M0LBs[thisrefid] >= dist - TH*dist:
            c_lb = getLB_oneQR(query, references[thisrefid], bboxes, dist)
            cluster_cals += 1
            if c_lb < dist:
#                dist2 = DTWdist[queryID][thisrefid]
                dist2 = DTW_a(query, references[thisrefid], W, dist)
                if dist > dist2:
                    dist = dist2
                    predId = thisrefid
            else:
                skip += 1
        else:
            dist2 = DTW_a(query, references[thisrefid], W, dist)
            #dist2 = DTWdist[queryID][thisrefid]
            if dist > dist2:
                dist = dist2
                predId = thisrefid
    end = time.time()
    coretime += (end - start)

    return dist, predId, skip, coretime, cluster_cals


def dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2], THs=[1]):
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

#    allTimes = []
    for idxset, dataset in enumerate(datasets):
        print(dataset + " Start!")
        assert (datasize[idxset] >= nqueries + nreferences)
        stuff = loadUCRData_norm_xs(datapath, dataset, nqueries + nreferences)
        size = len(stuff)
        length = stuff[0].shape[0]
        dim = min(stuff[0].shape[1], maxdim)
        print("Size: " + str(size))
        print("Dim: " + str(dim))
        print("Length: " + str(length))
        samplequery = stuff[:nqueries]
        samplereference = stuff[nqueries:nreferences + nqueries]
        # -------------------------------------------------
        if (nqueries * nreferences == 0):  # all series to be used
            qfrac = 0.3
            samplequery = stuff[:int(size * qfrac)]
            samplereference = stuff[int(size * qfrac):]
        # -------------------------------------------------

        print(dataset + ":  " + str(nqueries) + " queries, " + str(nreferences) + " references." +
              " Total dtw: " + str(nqueries * nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w) + "/"
            for K in Ks:
                for Q in Qs:
                    for TH in THs:
                        print("K="+str(K)+" Q="+str(Q)+" TH="+str(TH))
                        results = [DTWDistanceWindowLB_Ordered_LBPC_ (ids1, K, Q,
                                    query[ids1], reference, windowSize, TH) for ids1 in range(len(query))]
                        # if findErrors(dataset, maxdim, w, nqueries, nreferences, results, pathUCRResult):
                        #     print('Wrong Results!! Dataset: ' + dataset)
                        #     exit()
                        with open(toppath + str(nqueries) + "X" + str(
                                nreferences) + "_LBPC_K" + str(K) + "Q" + str(Q) + "TH"+ str(TH) + "_results.txt", 'w') as f:
                            for r in results:
                                f.write(str(r) + '\n')

    return 0


def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2], THs=[1], machineRatios=[1,1]):
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
        for K in Ks:
            for Q in Qs:
                for TH in THs:
                    results = readResultFile(
                        pathUCRResult + dataset + '/d' + str(maxdim) + "/w" + str(windows[0]) + "/" + str(nqueries) + "X" + str(
                            nreferences) + "_LBPC_K" + str(K) + "Q"+str(Q)+ "TH" + str(TH)+ "_results.txt")
                    tCore.append(sum(results[:, 3]))
                    skips.append(sum(results[:, 2]))
    tCore = np.array(tCore).reshape((ndatasets, -1))
    skips = np.array(skips).reshape((ndatasets, -1))

    tCorePlus = tCore
    #tDTW = np.tile(t1dtw, (skips.shape[1], 1)).transpose() * ((skips - totalPairs) * -1)
    tsum = rother * tCorePlus
    tsum_min = np.min(tsum, axis=1)
    setting_chosen = np.argmin(tsum,axis=1)
    skips_chosen = np.array( [skips[i,setting_chosen[i]] for i in range(skips.shape[0])] )
#    overhead = rother* np.array([tCorePlus[i,setting_chosen[i]] for i in range(tCorePlus.shape[0])])
    speedups = (rdtw * t1dtw[0:ndatasets] * NPairs) / tsum_min
#    overheadrate = overhead/(rdtw * t1dtw * NPairs)

    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBPC_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + '_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBPC_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + '_skipschosen.npy', skips_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBPC_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + '_settingchosen.npy', setting_chosen)
#    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
#            "_LBPC_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_overheadrate.npy', overheadrate)

    return 0

