import numpy as np
import glob


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

path='../Results/tryonesetting/'

maxdim_g = 5
nqueries_g = 0
nreferences_g = 0

fnm = path+'_AllDataSets/d'+str(maxdim_g)+'/'+ str(nqueries_g)+'X'+str(nreferences_g)
speedupFiles = glob.glob(fnm+"_*_speedups*.npy")
speedupFiles.sort()
speedupMethods = []
for f in speedupFiles:
    words = f.split('/')
    words = words[-1].split('_')
    if words[2]=='ws':
        speedupMethods.append(words[1]+words[2])
    else:
        speedupMethods.append(words[1])

skipFiles = glob.glob(fnm+'_*_skips*.npy')
skipFiles.sort()
skipMethods = []
for f in skipFiles:
    words = f.split('/')
    words = words[-1].split('_')
    if words[2]=='ws':
        skipMethods.append(words[1]+words[2])
    else:
        skipMethods.append(words[1])

# overheadFiles = glob.glob(fnm+'_*_overheadrate*.npy')
# overheadFiles.sort()
# overheadMethods = []
# for f in overheadFiles:
#     words = f.split('/')
#     words = words[-1].split('_')
#     if words[2]=='ws':
#         overheadMethods.append(words[1]+words[2])
#     else:
#         overheadMethods.append(words[1])

speedups = np.array([np.load(f) for f in speedupFiles])
skips = np.array([np.load(f) for f in skipFiles])
#overhead = np.array([np.load(f) for f in overheadFiles])
#alltable = np.array([speedups, skips, overhead])
with open(fnm+"_All_speedups.txt", 'w+') as f:
    for w in speedupMethods:
        f.write(w+',')
    f.write('\n')
    for r in speedups.transpose():
        for s in r:
            f.write("{:.2f},".format(s))
        f.write("\n")
#f=open(fnm+"_All_speedups.txt",'ab')
#np.savetxt(f, speedups.transpose(), delimiter=',')
#f.close()

with open(fnm+"_All_skips.txt", 'w+') as f:
    for w in skipMethods:
        f.write(w+',')
    f.write('\n')
    for r in skips.transpose():
        for s in r:
            f.write("{:.0f},".format(s))
        f.write("\n")

# with open(fnm+"_All_overhead.txt", 'w+') as f:
#     for w in overheadMethods:
#         f.write(w+',')
#     f.write('\n')
# f=open(fnm+"_All_overhead.txt",'ab')
# np.savetxt(f, overhead.transpose(), delimiter=',')
# f.close()

print('Done.')
