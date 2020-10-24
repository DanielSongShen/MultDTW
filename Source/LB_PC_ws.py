from Source.Util import *
import time


# This file implements X0 followed by X3 for cases that X0 fails but not miserably.
# It uses early abandoning on both X3 lower bounds calculations and DTW distance calculations
# X3: offline find bounding boxes; online compute the lower bounds when necessary.
# It use uniform grouping of bounding boxes: Every C consecutive windows share a set of bounding boxes.
# The dataCollection function saves the following:
#     the DTW distances, nearest neighbors, skips, coreTime, X3 invoked times in each individual directory: a text file


def findBoxes_onepoint (awindow, K, Q):
    '''
    Find the bounding boxes of one window
    :param awindow: an array of points
    :return: an array of boxes
    '''
    cellMembers = {}
    bboxes_oneref = []
    dims = len(awindow[0])

    overall_ls = [min(np.array(awindow)[:, idd]) for idd in range(dims)]
    overall_us = [max(np.array(awindow)[:, idd]) for idd in range(dims)]
    cells = [1 + int((overall_us[idd] - overall_ls[idd]) * Q) for idd in range(dims)]
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
        bboxes_oneref.append([l, u])
    if len(bboxes_oneref) > K:
        # combine all boxes except the first K-1 boxes
        sublist = bboxes_oneref[K - 1:]
        combinedL = [min([b[0][idd] for b in sublist]) for idd in range(dims)]
        combinedU = [max([b[1][idd] for b in sublist]) for idd in range(dims)]
        bboxes_oneref = bboxes_oneref[0:K - 1]
        bboxes_oneref.append([combinedL, combinedU])
    return bboxes_oneref

def getLB_oneQR_boxR (X, sl_bounds, dist, C):
    '''
    Get the lower bound of one series X to many references (others) based on the bounding boxes of the references.
    Early abandoning is used.
    :param X:
    :param others:
    :param dim:
    :param sl_bounds: an array of boxes. Every C points share one set of bounding boxes
              [ [[lows][highs]] [[lows][highs]] ...]
    :return:
    '''
    LB_sum = 0
    dim = len(X[0])
    slboundsOneY = sl_bounds
    for idx, x in enumerate(X):
        boxes = slboundsOneY[int(idx/C)]
        numBoxes = len(boxes)
        oneYbounds=[]
        for idbox in range(numBoxes):
            l = boxes[idbox][0]
            u = boxes[idbox][1]
            temp = math.sqrt(sum([(x[idd]-u[idd]) ** 2 if (x[idd] > u[idd]) else (l[idd]-x[idd])**2 if (x[idd] < l[idd]) else 0
                           for idd in range(dim)]))
            oneYbounds.append(temp)
        LB_sum+=min(oneYbounds)
        if LB_sum>=dist:
            return LB_sum
    return LB_sum

# def getLB_oneQR_boxR (Q, R, bboxes, dist):
#     '''
#     Get the lower bounds between two series, Q and R with multiple bounding boxes.
#     :param Q: A series.
#     :param R: A series.
#     :param bboxes: the bounding boxes of R.
#     :return: the lower bound between Q and R
#     '''
#     #  X and Y one series, is all references, dim is dimensions, sl_bounds has all the bounding boxes of all reference series
#     LB_sum = 0
#     dim = len(Q[0])
#     for idq, q in enumerate(Q):
#         numBoxes = len(bboxes[idq])
#         bounds=[]
#         for idbox in range(numBoxes):
#             l = bboxes[idq][idbox][0]
#             u = bboxes[idq][idbox][1]
#             temp = math.sqrt(sum([(q[idd]-u[idd]) ** 2 if (q[idd] > u[idd]) else (l[idd]-q[idd])**2
#                                     if (q[idd] < l[idd]) else 0 for idd in range(dim)]))
#             bounds.append(temp)
#         LB_sum+=min(bounds)
#         if LB_sum>=dist:
#             return LB_sum      # early abandoning
#     return LB_sum

def findBoundingBoxes_reuse_c (ref, K, W, Q, C):
    '''
    find the K bounding boxes for each window in ref with quantizations. uniformly grouped.
    :param ref: a data frame holding a reference serie
    :param K: the number of bounding boxes
    :param W: the window size
    :param Q: the number of cells in each dimension
    :param C: the num of windows to consider together in find bounding boxes
    :return: a dictionary allboxes_oneref = {boxes: [ [several(dim) lowends] [several(dim) highends] ],
        indices: [ index, index, ...]}  So, the boxes of point i on this ref are:
        allBoxes_oneref.boxes[allBoxes_oneref.indices[i]]
    '''
    length = ref.shape[0]
    indices = []
    bboxes = []
    Wu = W + C - 1
    # first point on ref

    # the rest
    for idx in range(0,length,C):
        awindow = ref[(idx - W if idx - W >= 0 else 0):(idx + Wu if idx + Wu <= length else length)]
        currentBoxes = findBoxes_onepoint(awindow, K, Q)
        bboxes.append(currentBoxes)
#    validatebboxes(ref, result, W)

    return bboxes


def getBoundingBoxes(references, w, K=4, Q=2, C=4):
    print("Bounding boxes finding Start!")
    start = time.time()
    bboxes = []
    for ir in range(len(references)):
#        print('Find boxes for reference '+str(ir))
        bboxes.append(findBoundingBoxes_reuse_c(np.array(references[ir]), K, w, Q, C))
    end = time.time()
    setuptime2003cluster_q = end - start
    print("Bounding boxes Done!")
    return bboxes, setuptime2003cluster_q

def DTWDistanceWindowLB_Ordered_X3rseac_ (X0LBs, bboxes, s1, refs, W, qid, C, TH=1):
    '''
    Compute the shortest DTW between a query and references series.
    :param queryID: the index number of this query
    :param X0LBs: the X0 lower bounds of the DTW between this query and each reference series
    :param bboxes: the precomputed boounding boxes
    :param K: the maximum number of clusters
    :param Q: the quantization level
    :param s1: the query
    :param refs: the references series
    :param W: the half window size
    :param C: the granunality in grouping bounding boxes
    :param TH: the triggering threshold of more expensive lower bound calculations
    :return: the DTW distance, the neareast neighbor of this query, the number of DTW distance calculations skipped,
             the number of times the clustering-based method is invoked
    '''
#    if qid==77:
#        print('qid==77')
    skip = 0
    cluster_cals = 0
    coretime = 0

    start = time.time()
    LBSortedIndex = np.argsort(X0LBs)
    #LBSortedIndex = sorted(range(len(X0LBs)),key=lambda x: X0LBs[x])
    predId = LBSortedIndex[0]
#    end = time.time()
#    coretime += (end - start)

    dist = DTW(s1, refs[predId], W)

#    start = time.time()
    for x in range(1, len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if X0LBs[thisrefid] >= dist:
            skip += (len(X0LBs) - x)
            break
        elif X0LBs[thisrefid] >= dist - TH*dist:
            c_lb = getLB_oneQR_boxR(s1, bboxes[thisrefid],dist, C)
            cluster_cals += 1
            if c_lb < dist:
                dist2 = DTW_a(s1, refs[thisrefid], W, dist)
                if dist > dist2:
                    dist = dist2
                    predId = thisrefid
            else:
                skip += 1
        else:
            dist2 = DTW_a(s1, refs[thisrefid], W, dist)
            if dist > dist2:
                dist = dist2
                predId = thisrefid
    end = time.time()
    coretime += (end - start)

    return dist, predId, skip, coretime, cluster_cals

def dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2], THs=[1], C=4):
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

    allsetupTimes = []
    allnboxes = []
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
            lb2003 = load_M0LBs(pathUCRResult,dataset,maxdim,w,nqueries,nreferences)

            for K in Ks:
                for Q in Qs:
                    for TH in THs:
                        print("K="+str(K)+" Q="+str(Q)+" TH="+str(TH))
                        if C<1:
                            TC = K
                        else:
                            TC = C
                        bboxes, setuptime = getBoundingBoxes(reference, w, K, Q, TC)
                        results = [DTWDistanceWindowLB_Ordered_X3rseac_ (lb2003[ids1], bboxes,
                                    query[ids1], reference, windowSize, ids1, TC, TH) for ids1 in range(len(query))]
                        nboxes = 0
                        for r in range(len(reference)):
                            uniqPointsBoxes = bboxes[r]
                            nboxes += sum([len(p) for p in uniqPointsBoxes])
                        with open(toppath + str(nqueries) + "X" + str(
                                nreferences) + "_LBPC_ws_K" + str(K) + "Q" + str(Q) + "TH"+ str(TH)+"C" + str(C) + "_results.txt", 'w') as f:
                            for r in results:
                                f.write(str(r) + '\n')
                        allsetupTimes.append(setuptime)
                        allnboxes.append(nboxes)
    np.save(pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/" + str(nqueries) + "X" + str(nreferences)
            + "_LBPC_ws_w" + intlist2str(windows) +"K" + intlist2str(Ks)+ "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + "C" + str(C) + "_setuptimes.npy", allsetupTimes)
    np.save(pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/" + str(nqueries) + "X" + str(nreferences)
            + "_LBPC_ws_w" + intlist2str(windows) +"K" + intlist2str(Ks)+ "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + "C" + str(C) + "_nboxes.npy", allnboxes)

    return 0

def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2], THs=[1], C=4, machineRatios=[1,1]):
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
    setupLBtimes = np.load(
        pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/" + str(nqueries) + "X" + str(nreferences)
        + "_LBMV_ws_w" + intlist2str(windows) + "_times.npy")
    X0tLB = setupLBtimes[:, 1]
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
    else:
        NPairs = [nqueries*nreferences for i in range(ndatasets)]
    ## -------------------

    for dataset in datasets:
        for K in Ks:
            for Q in Qs:
                for TH in THs:
                    results = readResultFile(
                        pathUCRResult + dataset + '/d' + str(maxdim) + "/w" + str(windows[0]) + "/" + str(nqueries) + "X" + str(
                            nreferences) + "_LBPC_ws_K" + str(K) + "Q"+str(Q)+ "TH" + str(TH) + "C" + str(C) + "_results.txt")
                    tCore.append(sum(results[:, 3]))
                    skips.append(sum(results[:, 2]))
    tCore = np.array(tCore).reshape((ndatasets, -1))
    skips = np.array(skips).reshape((ndatasets, -1))
#    tDTW = np.tile(t1dtw, (skips.shape[1], 1)).transpose() * ((skips - totalPairs) * -1)
#    tsum = rother * tCore + rdtw * tDTW
    tsum_min = np.min(tCore, axis=1)
    setting_chosen = np.argmin(tCore,axis=1)
    skips_chosen = np.array( [skips[i,setting_chosen[i]] for i in range(skips.shape[0])] )
    overhead = rother* (np.array([tCore[i,setting_chosen[i]] for i in range(tCore.shape[0])]) + X0tLB)
    speedups = (rdtw * t1dtw[0:ndatasets] * NPairs) / (rother*X0tLB + tsum_min)
#    overheadrate = overhead/(rdtw * t1dtw * NPairs)

    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBPC_ws_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + "C"+ str(C) + '_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBPC_ws_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + "C"+ str(C) +  '_skipschosen.npy', skips_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_LBPC_ws_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + "TH" + intlist2str(THs) + "C"+ str(C) +  '_settingchosen.npy', setting_chosen)
    #np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
    #        "_X3rsea_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_overheadrate.npy', overheadrate)

    return 0

