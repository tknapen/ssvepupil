#!/usr/bin/env python
# encoding: utf-8

"""@package Operators
This module offers various methods to process eye movement data

Created by Tomas Knapen on .
Copyright (c) 2010 TKLAB. All rights reserved.

More details.
"""
from __future__ import division

import os, pickle, inspect

import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as pl
import bottleneck as bn
import pandas as pd
import mne

import sklearn.decomposition.pca as pca
from sklearn import svm, grid_search

from scipy import ndimage, signal

from hedfpy import HDFEyeOperator
from hedfpy import EyeSignalOperator

from fir import FIRDeconvolution

from IPython import embed as shell 

sn.set(style="ticks")

class Pupil_SSVEP_Session(object):
	"""docstring for Pupil_SSVEP_Analyzer"""
	def __init__(self, analyzer, file_alias, trial_duration = 90.0, eye = 'R', stim_frequency = 2.0):
		super(Pupil_SSVEP_Session, self).__init__()
		self.analyzer = analyzer
		self.file_alias = file_alias
		self.trial_duration = trial_duration
		self.eye = eye
		self.stim_frequency = stim_frequency

	def assert_data_intern(self):
		if not hasattr(self, 'trial_times'):
			self.internalize_data()

	def internalize_data(self):
		""""""
			# load times per session:
		self.trial_times = self.analyzer.ho.read_session_data(self.file_alias, 'trials')
		self.trial_phase_times = self.analyzer.ho.read_session_data(self.file_alias, 'trial_phases')
		self.trial_parameters = self.analyzer.ho.read_session_data(self.file_alias, 'parameters')
		try:
			self.events = self.analyzer.ho.read_session_data(self.file_alias, 'events')
		except KeyError:
			self.events = []

		# check at what timestamps the recording started:
		self.session_start_EL_time = np.array( self.trial_phase_times[np.array(self.trial_phase_times['trial_phase_index'] == 1) * np.array(self.trial_phase_times['trial_phase_trial'] == 0)]['trial_phase_EL_timestamp'] )[0]
		self.session_stop_EL_time = np.array(self.trial_times['trial_end_EL_timestamp'])[-1]

		self.trial_indices = np.unique(self.trial_times['trial_start_index'])

		self.sample_rate = self.analyzer.ho.sample_rate_during_period([self.session_start_EL_time, self.session_stop_EL_time], self.file_alias)
		self.nr_samples_per_trial = int(self.trial_duration * self.sample_rate)
		self.from_zero_timepoints = np.linspace(0, 90, self.nr_samples_per_trial)

		# the signals to be analyzed

		self.pupil_bp_pt = np.array([np.array(self.analyzer.ho.signal_during_period(
							time_period = [self.trial_times[self.trial_times['trial_start_index']==i]['trial_start_EL_timestamp'], self.trial_times[self.trial_times['trial_end_index']==i]['trial_end_EL_timestamp']], 
							alias = self.file_alias, signal = 'pupil_bp_clean_zscore', requested_eye = 'R'))[-self.nr_samples_per_trial:] for i in range(len(self.trial_indices))]).squeeze() # 

		self.timestamps_pt = np.array([np.array(self.analyzer.ho.signal_during_period(
							time_period = [self.trial_times[self.trial_times['trial_start_index']==i]['trial_start_EL_timestamp'], self.trial_times[self.trial_times['trial_end_index']==i]['trial_end_EL_timestamp']], 
							alias = self.file_alias, signal = 'time', requested_eye = 'R'))[-self.nr_samples_per_trial:] for i in range(len(self.trial_indices))]).squeeze() 

		# replay features only instantaneous transitions
		# prepare for this
		if 'RP' in self.file_alias:
			self.colors = ['r','g']
			self.scancode_list = [39,41]
		else:
			self.colors = ['r','b','g']
			self.scancode_list = [89,90,91]

	def raw_signal_plot(self):
		self.assert_data_intern()

		f = pl.figure(figsize = (24,24))
		for x in range(len(self.trial_indices)):
			s = f.add_subplot(len(self.trial_indices), 1, x+1)
			pl.plot(self.timestamps_pt[x][::100], self.pupil_bp_pt[x][::100], 'k')
			if len(self.events) != 0:
				events_this_trial = self.events[(self.events['EL_timestamp'] > self.timestamps_pt[x][0]) & (self.events['EL_timestamp'] < self.timestamps_pt[x][-1])]
				for sc, scancode in enumerate(self.scancode_list):
					these_event_times = events_this_trial[events_this_trial['scancode'] == scancode]['EL_timestamp']
					for tet in these_event_times:
						pl.axvline(x = tet, c = self.colors[sc], lw = 5.0)
			sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.analyzer.fig_dir, self.file_alias + '_raw.pdf'))

	def behavioral_analysis(self):
		"""some analysis of the behavioral data, such as mean percept duration, 
		dominance ratio etc"""
		self.assert_data_intern()
		# only do anything if this is not a no report trial
		if 'RP' in self.file_alias:
			all_percepts_and_durations = [[],[]]
		else:
			all_percepts_and_durations = [[],[],[]]
		if not 'NR' in self.file_alias: #  and not 'RP' in self.file_alias
			for x in range(len(self.trial_indices)):
				if len(self.events) != 0:
					events_this_trial = self.events[(self.events['EL_timestamp'] > self.timestamps_pt[x][0]) & (self.events['EL_timestamp'] < self.timestamps_pt[x][-1])]
					for sc, scancode in enumerate(self.scancode_list):
						percept_start_indices = np.arange(len(events_this_trial))[np.array(events_this_trial['scancode'] == scancode)]
						percept_end_indices = percept_start_indices + 1
						
						# convert to times
						start_times = np.array(events_this_trial['EL_timestamp'])[percept_start_indices] - self.timestamps_pt[x,0]
						if len(start_times) > 0:
							if percept_end_indices[-1] == len(events_this_trial):
								end_times = np.array(events_this_trial['EL_timestamp'])[percept_end_indices[:-1]] - self.timestamps_pt[x,0]
								end_times = np.r_[end_times, len(self.from_zero_timepoints)]
							else:
								end_times = np.array(events_this_trial['EL_timestamp'])[percept_end_indices] - self.timestamps_pt[x,0]

							these_raw_event_times = np.array([start_times + self.timestamps_pt[x,0], end_times + self.timestamps_pt[x,0]]).T
							these_event_times = np.array([start_times, end_times]).T + x * self.trial_duration * self.sample_rate
							durations = np.diff(these_event_times, axis = -1)

							all_percepts_and_durations[sc].append(np.hstack((these_raw_event_times, these_event_times, durations)))

			self.all_percepts_and_durations = [np.vstack(apd) for apd in all_percepts_and_durations]

			# last element is duration, sum inclusive and exclusive of transitions
			total_percept_duration = np.concatenate([apd[:,-1] for apd in self.all_percepts_and_durations]).sum()
			total_percept_duration_excl = np.concatenate([apd[:,-1] for apd in [self.all_percepts_and_durations[0], self.all_percepts_and_durations[-1]]]).sum()

			self.ratio_transition = 1.0 - (total_percept_duration_excl / total_percept_duration)
			self.ratio_percept_red = self.all_percepts_and_durations[0][:,-1].sum() / total_percept_duration_excl

			self.red_durations = np.array([np.mean(self.all_percepts_and_durations[0][:,-1]), np.median(self.all_percepts_and_durations[0][:,-1])])
			self.green_durations = np.array([np.mean(self.all_percepts_and_durations[-1][:,-1]), np.median(self.all_percepts_and_durations[-1][:,-1])])
			self.transition_durations = np.array([np.mean(self.all_percepts_and_durations[1][:,-1]), np.median(self.all_percepts_and_durations[1][:,-1])])

			self.ratio_percept_red_durations = self.red_durations / (self.red_durations + self.green_durations)
			plot_mean_or_median = 0 # mean

			f = pl.figure(figsize = (8,4))
			s = f.add_subplot(111)
			for i in range(len(self.colors)):
				pl.hist(self.all_percepts_and_durations[i][:,-1], bins = 20, color = self.colors[i], histtype='step', lw = 3.0, alpha = 0.4, label = ['Red', 'Trans', 'Green'][i])
			pl.hist(np.concatenate([self.all_percepts_and_durations[0][:,-1], self.all_percepts_and_durations[-1][:,-1]]), bins = 20, color = 'k', histtype='step', lw = 3.0, alpha = 0.4, label = 'Percepts')
			pl.legend()
			s.set_xlabel('time [ms]')
			s.set_ylabel('count')
			sn.despine(offset=10)
			s.annotate("""ratio_transition: %1.2f, \nratio_percept_red: %1.2f, \nduration_red: %2.2f,\nduration_green: %2.2f, \nratio_percept_red_durations: %1.2f"""%(self.ratio_transition, self.ratio_percept_red, self.red_durations[plot_mean_or_median], self.green_durations[plot_mean_or_median], self.ratio_percept_red_durations[plot_mean_or_median]), (0.5,0.65), textcoords = 'figure fraction')
			pl.tight_layout()
			pl.savefig(os.path.join(self.analyzer.fig_dir, self.file_alias + '_dur_hist.pdf'))

	def tf_analysis(self, plot_Z = True, frequencies = None, vis_frequency_limits = [1.8, 2.2], nr_cycles = 16, analysis_sample_rate = 100):
		self.assert_data_intern()

		if frequencies == None:
			frequencies = np.linspace(1.0, self.analyzer.low_pass_pupil_f, 40)

		down_sample_factor = int(self.sample_rate/analysis_sample_rate)
		resampled_signal = self.pupil_bp_pt[:,::down_sample_factor]

		# complex tf results per trial
		self.tf_trials = mne.time_frequency.cwt_morlet(resampled_signal, analysis_sample_rate, frequencies, use_fft=True, n_cycles=nr_cycles, zero_mean=True)
		self.instant_power_trials = np.abs(self.tf_trials)

		# z-score power
		self.Z_tf_trials = np.zeros_like(self.instant_power_trials)
		m = self.instant_power_trials.mean(axis = -1)
		sd = self.instant_power_trials.std(axis = -1)

		for z in range(len(self.Z_tf_trials)):
			self.Z_tf_trials[z] = ((self.instant_power_trials[z].T - m[z]) / sd[z] ).T

		# some obvious conditions
		if plot_Z:
			tf_to_plot = self.Z_tf_trials
		else:
			tf_to_plot = self.instant_power_trials

		f = pl.figure(figsize = (24,24))
		for x in range(len(self.trial_indices)):
			s = f.add_subplot(len(self.trial_indices), 2, (x*2)+1)
			pl.imshow(np.squeeze(tf_to_plot[x,(frequencies > vis_frequency_limits[0]) & (frequencies < vis_frequency_limits[1]),::100]), cmap = 'seismic', extent = [self.from_zero_timepoints[0], self.from_zero_timepoints[-1], vis_frequency_limits[-1], vis_frequency_limits[0]], aspect='auto')
			sn.despine(offset=10)
			s = f.add_subplot(len(self.trial_indices), 2, (x*2)+2)
			# pl.imshow(np.squeeze(tf_to_plot[x,:,::100]), cmap = 'gray')
			pl.plot(self.from_zero_timepoints[::down_sample_factor], np.squeeze(np.squeeze(tf_to_plot[x,(frequencies > vis_frequency_limits[0]) & (frequencies < vis_frequency_limits[1]),:])).mean(axis = 0), 'k')
			if len(self.events) != 0:
				events_this_trial = self.events[(self.events['EL_timestamp'] > self.timestamps_pt[x][0]) & (self.events['EL_timestamp'] < self.timestamps_pt[x][-1])]
				for sc, scancode in enumerate(self.scancode_list):
					these_event_times = events_this_trial[events_this_trial['scancode'] == scancode]['EL_timestamp']
					for tet in these_event_times:
						pl.axvline(x = (tet - self.timestamps_pt[x,0]) / self.sample_rate, c = self.colors[sc], lw = 5.0)
			sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.analyzer.fig_dir, self.file_alias + '_%i_tfr.pdf'%nr_cycles))

		with pd.get_store(self.analyzer.h5_file) as h5_file: 
			for name, data in zip(['tf_complex_real', 'tf_complex_imag', 'tf_power', 'tf_power_Z'], 
				np.array([np.real(self.tf_trials), np.imag(self.tf_trials), self.instant_power_trials, self.Z_tf_trials], dtype = np.float64)):
				opd = pd.Panel(data, 
								items = pd.Series(self.trial_indices), 
								major_axis = pd.Series(frequencies), 
								minor_axis = self.from_zero_timepoints[::down_sample_factor])
				h5_file.put("/%s/tf/cycles_%s_%s"%(self.file_alias, nr_cycles, name), opd)

	def read_trans_counts(self):
		tc_file = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'number_of_percepts.txt')
		tcs = pd.read_csv(tc_file)
		if self.file_alias in list(tcs):
			self.trans_counts = np.array(tcs[self.file_alias])

	def project_phases(self, nr_cycles = 20, freqs_of_interest = [1.8, 2.2]):

		self.assert_data_intern()
		self.read_trans_counts()

		replay_phase = np.loadtxt(os.path.join(self.analyzer.sj_dir, 'phase_delay.txt'))[0]
		with pd.get_store(self.analyzer.h5_file) as h5_file: 
			real_data = h5_file.get("/%s/tf/cycles_%s_%s"%(self.file_alias, nr_cycles, 'tf_complex_real'))
			imag_data = h5_file.get("/%s/tf/cycles_%s_%s"%(self.file_alias, nr_cycles, 'tf_complex_imag'))
			power_data = h5_file.get("/%s/tf/cycles_%s_%s"%(self.file_alias, nr_cycles, 'tf_power_Z'))

		trial_numbers = np.array(real_data.keys())
		frequencies = np.array(real_data.major_axis)
		timepoints = np.array(real_data.minor_axis)

		real_m = np.array(real_data)[:,(frequencies>freqs_of_interest[0]) & (frequencies<freqs_of_interest[1]),:].mean(axis = 1)
		imag_m = np.array(imag_data)[:,(frequencies>freqs_of_interest[0]) & (frequencies<freqs_of_interest[1]),:].mean(axis = 1)
		power_m = np.array(power_data)[:,(frequencies>freqs_of_interest[0]) & (frequencies<freqs_of_interest[1]),:].mean(axis = 1)

		expected_phase_real = np.cos(replay_phase + timepoints * self.stim_frequency * 2.0 * np.pi)
		expected_phase_imag = np.sin(replay_phase + timepoints * self.stim_frequency * 2.0 * np.pi)

		complex_data = np.array([real_m, imag_m]).transpose((1,2,0))
		template_data = np.array([expected_phase_real, expected_phase_imag]).transpose((1,0))

		projected_data = np.zeros(complex_data.shape[:-1])
		for x in range(len(complex_data)):
			projected_data[x] = np.array([np.dot(c, t)/np.dot(t,t) for c, t in zip(complex_data[x], template_data)])
		
		# plot these timecourses per trial
		f = pl.figure(figsize = (8,24))
		for x in range(len(complex_data)):
			s = f.add_subplot(len(complex_data), 1, x+1)
			pl.plot(timepoints, projected_data[x], 'k', lw = 4.0)
			pl.plot(timepoints, power_m[x], 'k--', lw = 2.0)
			s.axhline(np.median(projected_data[x]), color = 'b', lw = 3.0)
			s.axhline(np.mean(projected_data[x]), color = 'r', lw = 3.0, ls = '--', alpha = 0.6)
			if hasattr(self, 'trans_counts'):
				s.annotate('%i'%self.trans_counts[x], (0.5,0.65), textcoords = 'axes fraction')
			s.set_ylim([-2,4])
			sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.analyzer.fig_dir, self.file_alias + '_projected.pdf'))

		# save out as dataframe
		pdf = pd.DataFrame(projected_data, index = trial_numbers, columns = timepoints)
		with pd.get_store(self.analyzer.h5_file) as h5_file: 
			h5_file.put("/%s/tf/cycles_%s_%s"%(self.file_alias, nr_cycles, 'projected'), pdf)

	def svm_classification(self, nr_cycles = 20, freqs_of_interest = [-np.inf, np.inf], n_components = 10):

		self.assert_data_intern()
		self.read_trans_counts()

		with pd.get_store(self.analyzer.h5_file) as h5_file: 
			power_data = h5_file.get("/%s/tf/cycles_%s_%s"%(self.file_alias, nr_cycles, 'tf_power_Z'))
		timepoints = np.array(power_data.minor_axis)

		# preprocess the timecourses, by PCA
		pd_r = np.array(power_data).transpose((1,0,2))
		pd_r = pd_r.reshape((np.array(power_data).shape[1], -1))

		p = pca.PCA(n_components = n_components)
		p.fit(pd_r.T)
		power_data_R = p.transform(pd_r.T)
		# power_data_R = power_data_R.reshape((n_components, len(self.trial_indices),power_data.shape[-1]))

		# time courses of behavior allow us to create labels for training and test sets
		self.behavioral_analysis()

		label_selection_array = np.zeros((2,pd_r.shape[-1]), dtype = bool)
		times_in_data = np.concatenate([timepoints*self.sample_rate + x*timepoints.max()*self.sample_rate for x in range(len(power_data))])

		if not hasattr(self, 'all_percepts_and_durations'): # the no report condition doesn't have percept definitions
			data_points = power_data_R
			label_selection_array = np.ones((2, power_data_R.shape[-1]))
			svc = None
		else:		# with knowledge on what happened when, we can train a decoder. 
			labels = []
			data_points = []
			for percept, i in zip([0,-1], [0,1]):
				event_times = self.all_percepts_and_durations[percept][:,[2,3]]
				for ev in event_times:
					label_selection_array[i] += (times_in_data > ev[0]) & (times_in_data < ev[1])
				labels.append(np.ones(pd_r.shape[-1])[label_selection_array[i]] * i)
				data_points.append(power_data_R[label_selection_array[i],:])

			labels = np.concatenate(labels)
			data_points = np.concatenate(data_points)

			gammas = np.logspace(-6, -1, 5)
			svc = svm.SVC(kernel='linear', probability = True)
			# clf = grid_search.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas), n_jobs=-1)
			svc.fit(data_points, labels)
		
		try: os.mkdir(os.path.join(self.analyzer.base_directory, 'svm'))
		except OSError: pass
		with open(os.path.join(self.analyzer.base_directory, 'svm', self.file_alias+'_svm.pickle'), 'w') as f:
			pickle.dump((svc, data_points, label_selection_array), f)
		return svc, data_points, label_selection_array



class Pupil_SSVEP_Analyzer(object):
	"""Pupil_SSVEP_Analyzer is a class that analyzes the results of a pupil size SSVEP experiment"""
	def __init__(self, sj_initial, base_directory, file_aliases, low_pass_pupil_f = 6, high_pass_pupil_f = 0.01):
		super(Pupil_SSVEP_Analyzer, self).__init__()

		self.sj_initial = sj_initial
		self.file_aliases = file_aliases
		self.base_directory = base_directory
		self.low_pass_pupil_f = low_pass_pupil_f
		self.high_pass_pupil_f = high_pass_pupil_f

		self.sj_dir = os.path.join(self.base_directory, self.sj_initial )
		self.fig_dir = os.path.join(self.base_directory, self.sj_initial, 'figs' )
		self.edf_files = [os.path.join(self.base_directory, 'raw', fa + '.edf') for fa in self.file_aliases]
		self.h5_file = os.path.join(self.base_directory, self.sj_initial, self.sj_initial + '.h5')


		try: 	os.mkdir(self.sj_dir)
		except OSError:		pass
		try: 	os.mkdir(self.fig_dir)
		except OSError:		pass

		os.chdir(self.base_directory)

		# initialize the hdfeyeoperator
		self.ho = HDFEyeOperator(self.h5_file)
		# insert the edf file contents only when the h5 is not present.
		if not os.path.isfile(self.h5_file):
			self.preprocess()

	def preprocess(self):
		# implicit preprocessing
		for i, ef in enumerate(self.edf_files):
			self.ho.add_edf_file(ef)
			self.ho.edf_message_data_to_hdf(alias = self.file_aliases[i])
			self.ho.edf_gaze_data_to_hdf(alias = self.file_aliases[i], pupil_hp = self.high_pass_pupil_f, pupil_lp = self.low_pass_pupil_f)

	def analyze(self, nr_cycles_tf = 12.0):
		for alias in self.file_aliases:
			pss = Pupil_SSVEP_Session(self, alias)
			# pss.raw_signal_plot()
			# pss.tf_analysis(nr_cycles = nr_cycles_tf)
			# pss.behavioral_analysis()
			pss.project_phases()
			pss.svm_classification()

	def analyze_tf(self, nr_cycles_tf = [40,20,12,8,4,2]):
		for alias in self.file_aliases:
			for nc_tf in nr_cycles_tf:
				pss = Pupil_SSVEP_Session(self, alias)
				pss.tf_analysis(nr_cycles = nc_tf)

	def get_experimental_phase(self, freqs_of_interest = [1.8, 2.2]):
		which_alias_reported_replay = [fa for fa in self.file_aliases if '_RP' in fa][0]

		with pd.get_store(self.h5_file) as h5_file: 
			replay_report_real = h5_file.get("/%s/tf/cycles_%s_%s"%(which_alias_reported_replay, 20, 'tf_complex_real'))
			replay_report_imag = h5_file.get("/%s/tf/cycles_%s_%s"%(which_alias_reported_replay, 20, 'tf_complex_imag'))

		frequencies = np.array(replay_report_real.major_axis)
		replay_report_real_m = np.array(replay_report_real, dtype = np.complex)[:,(frequencies>freqs_of_interest[0]) & (frequencies<freqs_of_interest[1]),:].mean(axis = 1)
		replay_report_imag_m = np.array(replay_report_imag)[:,(frequencies>freqs_of_interest[0]) & (frequencies<freqs_of_interest[1]),:].mean(axis = 1)

		replay_data = np.zeros(replay_report_real_m.shape, dtype = np.complex)
		replay_data.real = replay_report_real_m
		replay_data.imag = replay_report_imag_m

		angle_mean = np.angle(replay_data.mean(axis = 0).reshape([-1,100]).mean(axis = 0))
		real_mean = np.real(replay_data.mean(axis = 0).reshape([-1,100]).mean(axis = 0))
		imag_mean = np.imag(replay_data.mean(axis = 0).reshape([-1,100]).mean(axis = 0))

		distance_per_phase_real = np.array([(np.cos(np.linspace(0,4*np.pi,100, endpoint = False) + phase) - real_mean).max() for phase in np.linspace(0,2*np.pi,1000)])
		distance_per_phase_imag = np.array([(np.sin(np.linspace(0,4*np.pi,100, endpoint = False) + phase) - imag_mean).max() for phase in np.linspace(0,2*np.pi,1000)])
		phase_lag_real, phase_lag_imag = (np.linspace(0,2*np.pi,1000)[x] for x in (np.argmin(distance_per_phase_real), np.argmin(distance_per_phase_imag)))

		f = pl.figure()

		s = f.add_subplot(211)
		pl.plot(angle_mean, label = 'phase')
		pl.plot(real_mean, label = 'real')
		pl.plot(imag_mean, label = 'imag')

		pl.plot(np.arange(100),np.sin(np.linspace(0,4*np.pi,100, endpoint = False)), 'k--', label = 'sin')
		pl.plot(np.arange(100),np.cos(np.linspace(0,4*np.pi,100, endpoint = False)), 'k:', label = 'cos')
		pl.legend()

		s = f.add_subplot(212)

		pl.plot(np.linspace(0,2*np.pi,1000), distance_per_phase_real, 'k:', label = 'real')
		pl.plot(np.linspace(0,2*np.pi,1000), distance_per_phase_imag, 'k--', label = 'imag')
		s.axvline(phase_lag_real, color = 'r', lw = 2.0)
		s.axvline(phase_lag_imag, color = 'g', lw = 2.0)

		s.annotate('%0.3f'%phase_lag_imag, (0.5,0.3), textcoords = 'figure fraction')

		pl.legend()
		sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.fig_dir, 'phase_delay.pdf'))

		np.savetxt(os.path.join(self.sj_dir, 'phase_delay.txt'), np.array([phase_lag_real, np.degrees(phase_lag_real)]), delimiter = '\t', fmt = '%3.4f')

	def svm_decode(self, smoothing_widths = [0, 50, 100, 150, 200, 250, 300, 400, 500, 1000]):
		results = {}
		# train_alias = self.file_aliases[0]
		train_alias = [fa for fa in self.file_aliases if fa.split('_')[-1] == 'RP'][-1]

		# test_alias = [fa for fa in self.file_aliases if fa.split('_')[-1] == 'NR'][-1]
		test_alias = self.file_aliases[0]
		# shell()
		
		print 'train on ' + train_alias

		pss_train = Pupil_SSVEP_Session(self, train_alias)
		results.update({train_alias: pss_train.svm_classification()})

		print 'test on ' + test_alias

		pss_test = Pupil_SSVEP_Session(self, test_alias)
		pss_test.read_trans_counts()
		results.update({test_alias: pss_test.svm_classification()})

		replay_to_riv_prediction = results[train_alias][0].predict_proba(results[test_alias][1])

		rtr_ptr = replay_to_riv_prediction[:,0].reshape((8,-1))
		rtr_ptr_S = np.array([ndimage.gaussian_filter1d(rtr_ptr, sm, axis = -1, mode = 'constant', cval = 0.5) for sm in smoothing_widths])# 1 s smoothing width

		total_ratio_threshold = np.percentile(replay_to_riv_prediction, 100*pss_train.ratio_percept_red, interpolation = 'linear')
		total_duration_ratio_threshold = np.percentile(replay_to_riv_prediction, 100*pss_train.ratio_percept_red_durations[1], interpolation = 'linear')

		# plot these timecourses per trial
		f = pl.figure(figsize = (8,24))
		for x in range(len(rtr_ptr)):
			s = f.add_subplot(len(rtr_ptr), 1, x+1)
			pl.plot(np.linspace(0,pss_test.trial_duration,rtr_ptr.shape[-1]), rtr_ptr[x], 'k', lw = 2.0)
			for sm in range(rtr_ptr_S.shape[0]):
				pl.plot(np.linspace(0,pss_test.trial_duration,rtr_ptr.shape[-1]), rtr_ptr_S[sm,x], 'b--', lw = 2.0, alpha = 0.25 + 0.75 * (sm / rtr_ptr_S.shape[0]), label = '%i'%smoothing_widths[sm])
			s.axhline(total_ratio_threshold, color = 'g', ls = '--', lw = 2.0)
			s.axhline(total_duration_ratio_threshold, color = 'r', lw = 2.0, ls = '--', alpha = 0.6)
			if hasattr(pss_test, 'trans_counts'):
				s.annotate('%i'%pss_test.trans_counts[x], (0.05,0.1), textcoords = 'axes fraction', fontsize = 18)
			s.set_ylim([-0.2,1.1])
			pl.legend(fontsize = 8, ncol = len(rtr_ptr_S), loc = (0.0,-0.15))
			sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.fig_dir, self.sj_initial + '_svm_raw_%s_%s.pdf'%(train_alias, test_alias)))


		f = pl.figure(figsize = (8,24))
		for x in range(len(rtr_ptr)):
			s = f.add_subplot(len(rtr_ptr), 1, x+1)
			pl.imshow(rtr_ptr_S[:,x], cmap = 'seismic', extent = [0, pss_test.trial_duration, smoothing_widths[-1], smoothing_widths[0]], aspect='auto')
			if hasattr(pss_test, 'trans_counts'):
				s.annotate('%i'%pss_test.trans_counts[x], (0.05,0.1), textcoords = 'axes fraction', fontsize = 18)
			s.set_yticks([0, len(smoothing_widths)-1])
			s.set_yticklabels([smoothing_widths[0], smoothing_widths[0]])
			sn.despine(offset=10)
		pl.tight_layout()
		pl.savefig(os.path.join(self.fig_dir, self.sj_initial + '_svm_imshow_%s_%s.pdf'%(train_alias, test_alias)))

