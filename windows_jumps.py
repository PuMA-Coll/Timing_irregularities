#!/usr/bin/env python
import argparse
import os
import astropy.units as u
import matplotlib.pyplot as plt
import pint.fitter
import pint.residuals
import pint.toa
from pint.models import get_model, get_model_and_toas
import pint.logging
import pint.config
import numpy as np
import pint.residuals as res
import copy
from pint.models import BinaryELL1, BinaryDD, PhaseJump, parameter, get_model
from pint.simulation import make_fake_toas_uniform as mft
from astropy import units as u, constants as c
from uncertainties import ufloat
from uncertainties.umath import *
import pandas as pd
import astropy.stats as st
import logging
import sys
import matplotlib





#Filter ToAs: dot e inv are used inside mask_toas function
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


def delta_nu(nu0, nu1): #relative jump definition
        return (nu1-nu0)


def windows(parfile,timfile,fitf1=False,len_window=20,min_toas=5, thresh=1e-8): #Everything happens inside this function
	parfile = parfile #from parser
	timfile= timfile #from parser
	print("Remember your timfile must be ordered by MJD")
	m, t_all = get_model_and_toas(parfile, timfile) #import model and toas
	t=t_all[t_all.get_errors() < 1 * u.ms] #filtering toas with big errors
	m.F0.frozen=False #fit F0 = yes
	windows=t.get_mjds().value
	m.F1.frozen = not fitf1 #from parser we decide if we fit F1
	#We get from parfile F0, F1 and F2:
	window_first_MJD=[]
	F0_PEPOCH=[]
	F0_par=m.F0.value
	F0_par_error=m.F0.uncertainty_value
	F1_par=m.F1.value
	try:
		F2=m.F2.value
	except:
		F2=0
	PEPOCH=m.PEPOCH.value
	F0_PEPOCH_error=[]
	F1_PEPOCH=[]
	F1_PEPOCH_error=[]
	F0_PEPOCH_post=[]
	F0_PEPOCH_post_error=[]
	F1_PEPOCH_post=[]
	F1_PEPOCH_post_error=[]
	F0_jump=[]
	F1_jump=[]
	current_F0=m.F0.value
	current_F0_2=m.F0.value
	fail_w = 0
	MJD_chisq_pre=[]
	MJD_chisq_post=[]
	chisq_pre=[]
	chisq_post=[]
	F0_change=F1_par*(windows[1]-windows[0])*3600*24 #We use it to decide if better fit or not fit F1
	if F0_change<F0_par_error:
		print("Considering first window length, you probably should not fit F1")
	elif F0_change>F0_par_error:
		print("Considering first window length, you probably should fit F1")
	for i in range(int(windows[-1]-windows[0])):
		window_start = windows[0] + float(i)
		window_end = window_start + len_window
		if window_end <= windows[-1]:
			window = mask_toas(t, before=window_start, after=window_end)
            # Check if the window has at least 5 TOAs
			if len(window) >= min_toas:
				m.PEPOCH.value = window_start
				f = pint.fitter.DownhillWLSFitter(window, m)
				f.model["F0"].value = current_F0
				windows_err_1 = []
				windows_err_2=[]
				try:
					f.fit_toas(maxiter=10)
				except:
					windows_err_1.append(window_start)
					print("Couldn0t fit this one")
				MJD_chisq_pre.append(m.PEPOCH.value)
				chisq_pre.append(f.resids.chi2_reduced)
				current_F0 = f.model["F0"].value
				F0_PEPOCH.append(current_F0)
				F0_PEPOCH_error.append(f.model["F0"].uncertainty_value)
				F1_PEPOCH.append(f.model["F1"].value)
				F1_PEPOCH_error.append(f.model["F1"].uncertainty_value)
				m.PEPOCH.value = window_end
				f = pint.fitter.DownhillWLSFitter(window, m)
				f.model["F0"].value = current_F0_2
				try:
					f.fit_toas(maxiter=10)
				except:
					windows_err_2.append(window_end)
				MJD_chisq_post.append(m.PEPOCH.value)
				chisq_post.append(f.resids.chi2_reduced)
				current_F0_2 = f.model["F0"].value
				F0_PEPOCH_post.append(current_F0_2)
				F0_PEPOCH_post_error.append(f.model["F0"].uncertainty_value)
				F1_PEPOCH_post.append(f.model["F1"].value)
				F1_PEPOCH_post_error.append(f.model["F1"].uncertainty_value)
				window_first_MJD.append(window_start)
			elif len(window) < min_toas:
				fail_w = fail_w+1
				F0_PEPOCH.append(2)
				F0_PEPOCH_post.append(1)
				F1_PEPOCH.append(2)
				F1_PEPOCH_post.append(1)
				F0_PEPOCH_error.append(0)
				F1_PEPOCH_error.append(0)
				F0_PEPOCH_post_error.append(0)
				F1_PEPOCH_post_error.append(0)
				print("We skip the window" + str(window_start) + "-" + str(window_end))
	if fitf1==False:
		F1_error=np.zeros(len(F0_PEPOCH_error))
	for i in range(len(F0_PEPOCH)): #Set relative jump to 0 if window did not work
		if F0_PEPOCH[i]==2:
			if i >= len_window:
				F0_PEPOCH[i]=F0_PEPOCH_post[i-len_window]
				F1_PEPOCH[i]=F1_PEPOCH_post[i-len_window]
			else:
				F0_PEPOCH[i]=0
				F1_PEPOCH[i]=0
		if F0_PEPOCH_post[i]==1:
			if i < len(F0_PEPOCH)-len_window:
				F0_PEPOCH_post[i]=F0_PEPOCH[i+len_window]
				F1_PEPOCH_post[i]=F1_PEPOCH[i+len_window]
			else:
				F0_PEPOCH_post[i]=0
				F1_PEPOCH_post[i]=0
	for i in range(len(F0_PEPOCH)-(len_window)): #comparing two consecutive windows, we have #mjds - 2n jumps.
		#if F0_PEPOCH_post[i] != F0_PEPOCH[i+len_window]:
			#F0_PEPOCH_post[i]=F0_PEPOCH_post[i]+F1_PEPOCH_post[i]*3600*24+F2*(3600*24)**2/2
		nu0=ufloat(F0_PEPOCH_post[i], F0_PEPOCH_post_error[i])
		nudot0=ufloat(F1_PEPOCH_post[i], F1_PEPOCH_post_error[i])
		nu1=ufloat(F0_PEPOCH[i+len_window], F0_PEPOCH_error[i+len_window])
		nudot1=ufloat(F1_PEPOCH[i+len_window], F1_PEPOCH_error[i+len_window])
		F0_jump.append(delta_nu(nu0,nu1))
		F1_jump.append(delta_nu(nudot0, nudot1))
	no_glitches=0
	possible_detections=0
	day_of_detection_1=[]
	F1_detection_1=[]
	F1_detection_1_err=[]
	possible_glitches=[]
	hist_1=[]
	hist_1_err=[]
	windows_plot=window_first_MJD
	for i in range(len(F0_jump)):
		value=F0_jump[i].nominal_value
		sig=F0_jump[i].s
		sup_lim=value+sig
		inf_lim=value-sig
		if sup_lim < -thresh or inf_lim > thresh:
			hist_1.append(F0_jump[i].nominal_value) #for the histogram
			hist_1_err.append(F0_jump[i].s)
			possible_glitches.append(F0_jump[i])
			possible_detections+=1
			day_of_detection_1.append(windows[0]+len_window+float(i))
			F1_detection_1.append(F1_jump[i].nominal_value)
			F1_detection_1_err.append(F1_jump[i].s)
	hist_1_v2=[]
	hist_1_v2_err=[]
	dod_1_v2=[] #This means day_of_detection_v2
	F1_1_v2=[]
	F1_1_v2_err=[]
	j=0 #We work twice: for 1 sigma and 2 sigmas.
	if not hist_1:
		print("No glitches detected with 1 sigma")
	else:
		hist_1_v2.append(hist_1[0])
		hist_1_v2_err.append(hist_1_err[0])
		dod_1_v2.append(day_of_detection_1[0])
		F1_1_v2.append(F1_detection_1[0])
		F1_1_v2_err.append(F1_detection_1_err[0])
		for i in range(len(hist_1)-1):
			#Now we make sure detections are separated enough, and we keep only the biggest one when we detect many times the same glitch
			if (day_of_detection_1[i+1]- dod_1_v2[-1]) > len_window:
				j=j+1
				hist_1_v2.append(hist_1[i+1])
				hist_1_v2_err.append(hist_1_err[i+1])
				dod_1_v2.append(day_of_detection_1[i+1])
				F1_1_v2.append(F1_detection_1[i+1])
				F1_1_v2_err.append(F1_detection_1_err[i+1])
			elif (day_of_detection_1[i+1]- dod_1_v2[-1]) < len_window and np.abs(hist_1[i+1])>np.abs(hist_1_v2[-1]):
				hist_1_v2[j]=hist_1[i+1]
				hist_1_v2_err[j]=hist_1_err[i+1]
				dod_1_v2[j]=day_of_detection_1[i+1]
				F1_1_v2[j]=F1_detection_1[i+1]
				F1_1_v2_err[j]=F1_detection_1_err[i+1]

	print("Possible detections with 1 sigma=  "+str(len(dod_1_v2)))
	for i in range(len(dod_1_v2)):
		print("Possible detection on " + str(dod_1_v2[i]) + " of " + str(hist_1_v2[i]) + " and " + "F1 jump= " + str(F1_1_v2[i]))

	if windows_err_1:
		print("From before the window, I couldn't fit ToAs for: ",windows_err_1)
	if windows_err_2:
		print("From after the window, I couldn't fit ToAs for: ",windows_err_2)
	return np.array(dod_1_v2), np.array(hist_1_v2), np.array(hist_1_v2_err), np.array(F1_1_v2), np.array(F1_1_v2_err), np.array(MJD_chisq_pre), np.array(chisq_pre), np.array(MJD_chisq_post), np.array(chisq_post)



def set_argparse():
   # add arguments
   parser = argparse.ArgumentParser(prog='windows_days_new_plot.py',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description='Main pipeline for mini glitch detections')
   parser.add_argument('--min_toas', default=5, type=int,
      help='minimun number of observations in each window')
   parser.add_argument('--len_window', default=20, type=int,
      help='length of each window')
   parser.add_argument('--thresh', default=1e-8, type=float,
      help='relative jump of glitch alert')
   parser.add_argument('--fitf1', default=False, type=lambda x: (str(x).lower() == 'true'), help='Fit F1? False or True')
   parser.add_argument('--parfile', help='name of the file.par')
   parser.add_argument('--timfile', help='name of the file.tim')
   return parser.parse_args()


if __name__ == '__main__':
	# get cli-arguments
	pint.logging.setup(level="ERROR")
	args = set_argparse()
	result=windows(args.parfile,args.timfile,args.fitf1, args.len_window, args.min_toas, args.thresh)
	df1=pd.DataFrame()
	df1["dod_1_v2"]=result[0]
	df1["hist_1_v2"]=result[1]
	df1["hist_1_v2_err"]=result[2]
	df1["F1_1_v2"]=result[3]
	df1["F1_1_v2_err"]=result[4]
	df1.to_csv('detections_len-'+str(args.len_window)+'_fitf1-'+str(args.fitf1)+'.csv', index=False)
	df2=pd.DataFrame()
	df2["MJD_chisq_pre"]=result[5]
	df2["chisq_pre"]=result[6]
	df2["MJD_chisq_post"]=result[7]
	df2["chisq_post"]=result[8]
	df2.to_csv('chisq_len-'+str(args.len_window)+'_fitf1-'+str(args.fitf1)+'.csv', index=False)
