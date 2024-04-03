import numpy as np
import os

import astropy.units as u

# This will change which output method matplotlib uses and may behave better on some machines
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pint.fitter
import pint.residuals
import pint.toa
from pint.toa import get_TOAs
from pint.models import get_model, get_model_and_toas
import pint.logging
from pint.models import PhaseJump
import pandas as pd
import argparse
# setup logging

# %%
import pint.config

def dot(l1,l2):
    return np.array([v1 and v2 for v1,v2 in zip(l1,l2)])
def inv(l):
    return np.array([not i for i in l])

def mask_toas(toas,before=None,after=None,on=None,window=None): #This function filters MJD between before and after
    cnd=np.array([True for t in toas.get_mjds()])
    if before is not None:
        cnd = dot(cnd,toas.get_mjds().value >= before)
    if after is not None:
        cnd = dot(cnd,toas.get_mjds().value < after)
    if on is not None:
        on=np.array(on)
        for i,m in enumerate(on):
            m=m*u.day
            if type(m) is int:
                cnd = dot(cnd,inv(np.abs(toas.get_mjds()-m).astype(int) == np.abs((toas.get_mjds()-m)).min().astype(int)))
            else:
                cnd = dot(cnd,inv(np.abs(toas.get_mjds()-m) == np.abs((toas.get_mjds()-m)).min()))
    if window is not None:
        if len(window)!=2:
            raise ValueError("window must be a 2 element list/array")
        window = window*u.day
        lower = window[0]
        upper = window[1]
        cnd = dot(cnd,toas.get_mjds() < lower)+dot(cnd,toas.get_mjds() > upper)
    #print(f'{sum(cnd)}/{len(cnd)} TOAs selected')
    return toas[cnd]





def plot_glitches(parfile,timfile,fitf1,len_window,thresh,sigmas,detections):
	#Section: plot all individually
	df=pd.read_csv(detections, sep=',', header=0)
	detections_sigmas=df[np.abs(df['hist_1_v2']/df['hist_1_v2_err'])>sigmas]
	ds=detections_sigmas
	m, t_all = get_model_and_toas(parfile, timfile) #import model and toas
	t=t_all[t_all.get_errors() < 1 * u.ms] #filtering toas with big errors
	m.F0.frozen=False #fit F0 = yes
	windows=t.get_mjds().value
	m.F1.frozen = not fitf1 #from parser we decide if we fit F1
	dod_1_v2=np.array(ds["dod_1_v2"])
	hist_1_v2=np.array(ds["hist_1_v2"])
	hist_1_v2_err=np.array(ds["hist_1_v2_err"])
	F1_1_v2=np.array(ds["F1_1_v2"])
	F1_1_v2_err=np.array(ds["F1_1_v2_err"])
	for i in range(len(dod_1_v2)):
		print("Possible detection on " + str(dod_1_v2[i]) + " of " + str(hist_1_v2[i]) + " and " + "F1 jump= " + str(F1_1_v2[i]))
		m.PEPOCH.value=dod_1_v2[i]
		window = mask_toas(t, before=m.PEPOCH.value-len_window, after=m.PEPOCH.value) #Get F0 again in the pre-glitch window
		f = pint.fitter.DownhillWLSFitter(window, m)
		f.fit_toas(maxiter=100)
		m.F0.value = f.model["F0"].value
		m.F1.value = f.model["F1"].value
        	#Now I plot the whole window
		window = mask_toas(t, before=m.PEPOCH.value-len_window, after=m.PEPOCH.value+len_window)
		rs=pint.residuals.Residuals(window, m)
		plt.figure()
		plt.errorbar(window.get_mjds().value, rs.time_resids.to(u.ms), rs.toas.get_errors().to(u.ms), label="Residuals", fmt="X",markersize=1)
		plt.axvline(x=m.PEPOCH.value,color="black",linestyle='--',label='Possible glitch at '+str(m.PEPOCH.value))
		plt.legend()
		plt.grid()
		plt.xlabel("MJD")
		plt.ylabel("Residuals (ms)")
		plt.savefig("tg_"+str(m.PEPOCH.value)+".png")

		plt.clf()
		plt.close()
	#Section: plot all
	m, t_all = get_model_and_toas(parfile, timfile) #import model and toas
	t=t_all[t_all.get_errors() < 1 * u.ms] #filtering toas with big errors
	rs = pint.residuals.Residuals(t, m)
	fig, ax1 = plt.subplots(1,1)
	ax1.errorbar(rs.toas.get_mjds(),rs.time_resids.to(u.ms),rs.toas.get_errors().to(u.ms), fmt=".", label='A1+A2', color="blue")
	ax3=ax1.twinx()
	ax3.errorbar(ds['dod_1_v2'][ds['hist_1_v2']>0], ds['hist_1_v2'][ds['hist_1_v2']>0], yerr=ds['hist_1_v2_err'][ds['hist_1_v2']>0], fmt="x", label='candidate G', color='green')
	ax3.errorbar(ds['dod_1_v2'][ds['hist_1_v2']<0], ds['hist_1_v2'][ds['hist_1_v2']<0], yerr=ds['hist_1_v2_err'][ds['hist_1_v2']<0], fmt="x", label='candidate AG', color='red')
	ax3.errorbar(ds['dod_1_v2'], ds['F1_1_v2']*1e-6, yerr=ds['F1_1_v2_err']*1e-6, markerfacecolor='white', markeredgecolor='orange',fmt="o", label='F1 jump')
	ax3.set_yscale("symlog")
	ax1.legend()
	ax1.set_xlabel("MJD")
	ax1.set_ylabel("Residuals(ms)")
	ax1.grid()
	ax3.legend()
	ax3.set_ylabel(r'$\Delta \nu / \nu$')
	ax3.grid()
	plt.show()

def set_argparse():
   # add arguments
   parser = argparse.ArgumentParser(prog='plot_jumps.py',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description='Main pipeline for mini glitch detections')
   parser.add_argument('--sigmas', default=1, type=int,
      help='sigmas requiered for the detections')
   parser.add_argument('--len_window', default=24, type=int,
      help='length of each window')
   parser.add_argument('--thresh', default=0, type=float,
      help='relative jump of glitch alert')
   parser.add_argument('--fitf1', default=True, type=lambda x: (str(x).lower() == 'true'), help='Fit F1? False or True')
   parser.add_argument('--parfile', help='name of the file.par')
   parser.add_argument('--timfile', help='name of the file.tim')
   parser.add_argument('--detections', help='File with results')
   return parser.parse_args()


if __name__ == '__main__':
        # get cli-arguments
        pint.logging.setup(level="ERROR")
        args = set_argparse()
        plot_glitches=plot_glitches(args.parfile,args.timfile,args.fitf1, args.len_window, args.thresh, args.sigmas, args.detections)
