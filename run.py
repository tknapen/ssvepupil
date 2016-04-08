#!/usr/bin/env python
# encoding: utf-8

"""@package Operators
This module offers various methods to process eye movement data

Created by Tomas Knapen on .
Copyright (c) 2010 TKLAB. All rights reserved.

More details.
"""
from __future__ import division

import os 
from joblib import Parallel, delayed

from initial import Pupil_SSVEP_Analyzer

# define subject and run information

subject_file_aliases = {
	# 'JZ': ['JZ_1', 'JZ_1_RP', 'JZ_1_NR', 'JZ_1_NRP'],
	'TK': [  'TK_1', 'TK_1_RP', 'TK_1_NR', 'TK_1_NRP', 'TK_2_NR' ], #
	# 'JD': [  'JD_1', 'JD_1_RP', 'JD_1_NR', 'JD_1_NRP', 'JD_2' ], #
	# 'PZ': [  'PZ_1', 'PZ_1_NR'],
	# 'PZ': [  'PZ_2'],
}

BASE_DIR = '/Users/knapen/Documents/grants/NWO-CAS_2016/supporting_pilot_exp/data'

# analyze single subject run function
def run_subject(subject_id, base_dir, file_aliases):

	psa = Pupil_SSVEP_Analyzer(sj_initial = subject_id, base_directory = base_dir, file_aliases = file_aliases)
	# not implementing , low_pass_pupil_f = low_pass_pupil_f, high_pass_pupil_f = high_pass_pupil_f right now

	psa.analyze_tf()
	psa.get_experimental_phase()
	psa.analyze()
	psa.svm_decode()


	return True

# analyze subjects in parallel
def analyze_subjects(sjs, parallel = True ):
	os.chdir('../data/')

	if len(sjs) > 1 and parallel: 
		# parallel processing with joblib
		res = Parallel(n_jobs = 24, verbose = 9)(delayed(run_subject)(k, BASE_DIR, subject_file_aliases[k]) for k in subject_file_aliases.keys())
	else:
		for k in subject_file_aliases.keys():
			run_subject(k, BASE_DIR, subject_file_aliases[k])


#####################################################
#	main
#####################################################

def main():
	analyze_subjects(subject_file_aliases, parallel = True)

if __name__ == '__main__':
	main()