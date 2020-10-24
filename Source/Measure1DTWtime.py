from Source.Util import *

# Measure the DTW time that can be used as the basis for deriving the ultimate speedups

def MeasurePrimeTimes(resultpath, datasetsNameFile, datasetsSizeFile, datapath, maxdim, windowsize, Ks_g, Qs_g, nqueries=3, nreferences=3):
    outputdir = resultpath+'_AllDataSets/d'+ str(maxdim) + '/'
    t1dtwfile = resultpath + '_AllDataSets/d' + str(maxdim) + '/' + 'Any_Anyw' + str(windowsize) + '_t1dtw.npy'
    datasets = []

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize = []
    with open(datasetsSizeFile, 'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

#    datasets = ["ArticularyWordRecognition", "AtrialFibrillation"]

    alltimes=[]
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

        print(dataset + ":  " + str(nqueries) + " queries, " + str(nreferences) + " references." +
              " Total dtw: " + str(nqueries * nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        templist = []
        totalpairs = len(query)*len(reference)
        # measure the times of DTW
        start = time.time()
        dtw = [DTW(s1, s2, windowsize) for s2 in reference for s1 in query]
        end = time.time()
        timedtw = (end - start)/totalpairs
        alltimes.append(timedtw)

    np.save(t1dtwfile, np.array(alltimes))
