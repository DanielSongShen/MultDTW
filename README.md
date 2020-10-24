# MultivariteDTW
# Author: Daniel Shen.
# Date: June 2020

This package contains several methods on conducting multivariate DTW-NN.

====== Content of Folders ===============
Data: the pickled datasets for processing.
Results: the folder to store the produced results.
Source: the source code folder.

====== List of Methods ==================
LB_MV_ws: min and max are precomputed on references (ws stands for with setup).
LB_MV: online version. No pre-setup.

LB_TI_ws: LBMV followed by online LB_TI with pre-setup.
LB_TI: online version. No pre-setup.

LB_PC_ws: point clustering-based method with pre-setup.
LB_PC: point clustering-based method. No pre-setup.

===== How to run ========================
>>>> Example 1:
Step 1) Go to Source folder.
Step 2) Run
    test_onesetting.py 2
  It will invoke all the methods on the second dataset in 'Data/' folder,
     produce the performance measurement results, and put them into the folder 'Results'.
Step 3) Run
    peakResults.py
  It will generate the following files in folder ../Results/tryonesetting/
    0X0_All_speedups.txt: contains the speedups from each of the methods.
    0X0_All_skips.txt: contains the number of skips from each of the methods.

>>>> Example 2:
Step 1) Go to Source folder.
Step 2) Run
    test_tryall.py 2
  It will invoke all the methods on the second dataset in 'Data/' folder and try every setting for a method,
     produce the performance measurement results, and put them into the folder 'Results'.
Step 3) Run
    peakResults.py
  It will generate the following files in folder ../Results/tryallsettings/
    0X0_All_speedups.txt: contains the speedups from each of the methods.
    0X0_All_skips.txt: contains the number of skips from each of the methods.
  Which includes the best results of each method for each datasest.

===== Notes =============================
* In the real deployment, parameters of TI and PC methods would need to be selected
  through a separate process by experimenting with the candidate series. Similarly, for a given dataset, one can use a small set of samples to pick the better choice between LM_TI and LM_PC; this adaptive scheme leads to a method named TC-DTW.


* The included datasets were part of the UCR multivariate datasets:
   http://www.timeseriesclassification.com/index.php
  More can be downloaded from that website. Picked files are used by this package.

* If an error like module 'Source' was not found appears when running the commands, just add
  the root directory of this package (i.e., the path of NewMDTW) to your PYTHONPATH environment
  variable.