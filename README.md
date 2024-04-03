# Timing_irregularities
Codes and data realease corresponding to "Timing irregularities in the glitching pulsar monitoring campaign at the IAR"

About the data release, we present the .par and .tim that we used for our work. We also present as "JXXXX-XXXX_glitch.par" the characterization of the three giant glitches with their respectives recovery terms.

"windows_jumps.py" is a PINT-based algorithm used to look for irregularities in timing data. It calculates F0 and F1 jump for two consecutive windows of L days, if there is a minimum of N_min ToAs inside each and report significant changes in F0. Thresh is the minimum size of the detections that we want to report, we set it to 0.


The command is the following:

python windows_jumps.py --min_toas MIN_TOAS   --len_window LEN_WINDOW  --thresh THRESH  --fitf1 FITF1(Bool) --parfile PARFILE --timfile TIMFILE

The output is file1.csv (detections) and file2.csv (chisq of all the fitting processess).

There is another code to plot the residuals of the detections:

python plot_jumps.py   --sigmas SIGMAS  (sigmas requiered for the detections, default: 1) --len_window LEN_WINDOW --thresh THRESH --fitf1 FITF1 --parfile PARFILE --timfile TIMFILE --detections file1.csv

Please refer to our paper in case of use.
