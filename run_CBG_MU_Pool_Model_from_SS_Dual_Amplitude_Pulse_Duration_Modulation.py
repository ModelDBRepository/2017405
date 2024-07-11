# -*- coding: utf-8 -*-
"""
Created on April 03 14:27:26 2020

Description: Motor network model of the coupled cortico-basal ganglia network and motoneuron pool 
			 implemented in PyNN using the simulator Neuron. The model simulates the STN LFP, DBS electric field, 
			 and FDI muscle force and EMG signals. This version of the model loads the steady 
			 state of the model from a previous initial run of the model which integrating the model to 
			 the steady state. The model is then integrated to simulate a 30 s period of tremor activity
			 followed by a 30 s period of elevated beta activity. Increased synaptic drive to the motoneuron
			 pool is simulated during the elevated beta period to simulate muscle activation during voluntary
			 movement.
			 
@author: John Fleming, john.fleming@ucdconnect.ie
"""

import neuron
h = neuron.h

from pyNN.neuron import setup, run, reset, run_until, run_to_steady_state, run_from_steady_state, end, simulator, Population, SpikeSourcePoisson, SpikeSourceArray, Projection, OneToOneConnector, AllToAllConnector, FromFileConnector, FixedNumberPreConnector, CloneConnector, StaticSynapse, TsodyksMarkramSynapse, NativeCellType, SpikeSourcePoisson, SpikeSourceArray, DCSource, NoisyCurrentSource, StepCurrentSource
from pyNN.random import RandomDistribution, NumpyRNG
from pyNN import space
from Cortical_Basal_Ganglia_Cell_Classes import Cortical_Neuron_Type, Interneuron_Type, STN_Neuron_Type, GP_Neuron_Type, Thalamic_Neuron_Type
from Depressing_Spinal_Motoneuron_Cell_Classes import Spinal_Motoneuron_Synaptic_Bombardment_Point_Process_Type, motoneuron_pool_parameter_bounds, generate_motoneuron_population_parameters
from Electrode_Distances import distances_to_electrode, collateral_distances_to_electrode
from pyNN.parameters import Sequence
from Controllers import Controller_Scheduler, State_Observer, standard_PID_Controller, standard_PID_Controller_Plus_Bias, Constant_Controller
import random
import neo.io
import quantities as pq
import numpy as np
import math
from scipy import signal, interpolate
import os
import sys

# Import global variables for GPe DBS
import Global_Variables as GV

def generate_poisson_spike_times(pop_size, start_time, duration, fr, timestep, random_seed):
	""" generate_poisson_spike_times generates (N = pop_size) poisson distributed spiketrains
		with firing rate fr.
	
	Example inputs:
		pop_size = 10			
		start_time = 0.0		# ms
		end_time = 6000.0		# ms
		timestep = 1  			# ms
		fr = 1					# Hz
	"""
	
	# Convert to sec for calculating the spikes matrix
	dt = float(timestep)/1000.0                         # sec
	tSim = float(((start_time+duration) - start_time)/1000.0)         # sec
	nBins = int(np.floor(tSim/dt))

	spikeMat = np.where(np.random.uniform(0,1,(pop_size, nBins)) < fr*dt)

	# Create time vector - ms
	tVec = np.arange(start_time, start_time+duration, timestep)

	# Make array of spike times
	for neuron_index in np.arange(pop_size):
		neuron_spike_times = tVec[spikeMat[1][np.where(spikeMat[0][:]==neuron_index)]]
		if neuron_index == 0:
			spike_times = Sequence(neuron_spike_times)
		else:
			spike_times = np.vstack((spike_times, Sequence(neuron_spike_times)))

	poisson_spike_times = spike_times
	return poisson_spike_times
	
def generate_Noise_Signal(start_time, stop_time, dt, mean_amplitude, noise_stdev):
	"""Generate noise current for simulating noisy inputs to the dendrites
	
	Example inputs:
		start_time = 0				# ms
		stop_time = 20000			# ms
		dt = 0.1					# ms 
		mean_amplitude = 1.0			# nA
	"""
	
	# Generate the noise signal
	temp_num_t = int(round((stop_time - start_time) / dt))
	times = start_time + dt * np.arange(temp_num_t)
	times = np.append(times, stop_time)
	
	# Generate ramp base constant value signal
	mean_noise_signal = mean_amplitude*np.ones(len(times))
	
	# Generate the noise component of the signal
	noise_signal = noise_stdev * np.random.randn(len(times))
	
	# Generate low pass filter coefficients - design filter in s/Hz (not ms)
	Fs = (1.0/dt)*1e3
	nyq = 0.5*Fs
	lowcut = 150.0/nyq
	N = 5
	rp = 0.5
	b, a = signal.cheby1(N, rp, lowcut, 'low')
	
	# lowpass filter the noise
	filtered_noise_signal = signal.filtfilt(b, a, noise_signal)
	
	# constant mean noise signal
	noise_current = mean_noise_signal + filtered_noise_signal
	
	return times, noise_current

def generate_Ramped_Signal(start_time, stop_time, dt, ramp_duration, max_amplitude):
	"""Generate ramped bias current
	
	Example inputs:
		start_time = 0				# ms
		stop_time = 20000			# ms
		dt = 0.1					# ms 
		ramp_duration = 6000.0      # ms
		max_amplitude = 1.0		# nA
	"""
	
	temp_num_t = int(round((stop_time - start_time) / dt))
	times = start_time + dt * np.arange(temp_num_t)
	times = np.append(times, stop_time)
	
	# Generate ramp base constant value signal
	ramp_signal = max_amplitude*np.ones(len(times))
	
	# Ramp the noise amplitude (mean and stdev values)
	ramp_indices = np.where(times<=ramp_duration)[0]
	ramp_function = ramp_indices/np.max(ramp_indices)
	    
	ramp_signal[ramp_indices] = ramp_function * ramp_signal[ramp_indices]
	
	return times, ramp_signal
	
def generate_Ramped_Noise_Signal(start_time, stop_time, dt, ramp_duration, max_amplitude, noise_stdev):
	"""Generate noise current for simulating noisy inputs to the dendrites
	
	Example inputs:
		start_time = 0				# ms
		stop_time = 20000			# ms
		dt = 0.1					# ms 
		ramp_duration = 6000.0      # ms
		max_amplitude = 1.0		# nA
	"""
	
	# Generate the ramped signal
	times, ramp_signal = generate_Ramped_Signal(start_time, stop_time, dt, ramp_duration, max_amplitude)
	
	# Generate the noise component of the signal
	noise_signal = noise_stdev * np.random.randn(len(times))
	
	# Generate low pass filter coefficients - design filter in s/Hz (not ms)
	Fs = (1.0/dt)*1e3
	nyq = 0.5*Fs
	lowcut = 150.0/nyq
	N = 5
	rp = 0.5
	b, a = signal.cheby1(N, rp, lowcut, 'low')
	
	# lowpass filter the noise
	filtered_noise_signal = signal.filtfilt(b, a, noise_signal)
	
	# ramped noise signal
	ramped_noise_current = ramp_signal + filtered_noise_signal
	
	return times, ramped_noise_current
	
def generate_DBS_Signal(start_time, stop_time, dt, amplitude, frequency, pulse_width, offset):
	"""Generate monophasic square-wave DBS signal
	
	Example inputs:
		start_time = 0				# ms
		stop_time = 12000			# ms
		dt = 0.01					# ms 
		amplitude = -1.0			# mA - (amplitude<0 = cathodic stimulation, amplitude>0 = anodic stimulation)
		frequency = 130.0			# Hz
		pulse_width	= 0.06			# ms
		offset = 0					# mA
	"""
	
	times = np.round(np.arange(start_time, stop_time, dt), 2)
	tmp = np.arange(0, stop_time - start_time, dt)/1000.0
		
	# Calculate the duty cycle of the DBS signal
	T = (1.0/frequency)*1000.0	# time is in ms, so *1000 is conversion to ms
	duty_cycle = ((pulse_width)/T)
	#DBS_Signal = offset + amplitude * (1.0+signal.square(2.0 * np.pi * frequency * tmp, duty=duty_cycle))/2.0
	DBS_Signal = offset + 1.0 * (1.0+signal.square(2.0 * np.pi * frequency * tmp, duty=duty_cycle))/2.0 		# Need to initially set value > 0 to find last spike time, then can scale by amplitude
	DBS_Signal[-1] = 0.0
	
	# Calculate the time for the first pulse of the next segment
	last_pulse_index = np.where(np.diff(DBS_Signal)<0)[0][-1]
	next_pulse_time = times[last_pulse_index] + T - pulse_width
	
	# Rescale amplitude
	DBS_Signal *= amplitude
	
	return DBS_Signal, times, next_pulse_time
	
def make_beta_cheby1_filter(Fs, N, rp, low, high):
	"""Calculate bandpass filter coefficients (1st Order Chebyshev Filter)"""
	nyq = 0.5*Fs
	lowcut = low/nyq
	highcut = high/nyq

	b, a = signal.cheby1(N, rp, [lowcut, highcut], 'band')
	
	return b, a
	
def calculate_beta_ARV(lfp_signal, tail_length, beta_b, beta_a):
	"""Calculate the average rectified value of the beta-band for the current LFP signal window, i.e. beta Average Rectified Value (ARV)
	
	Exaqmple inputs:
		lfp_signal 			- window of LFP signal												# samples
		tail_length 		- tail length which will be discarded due to filtering artifact		# samples
		beta_b, beta_a 		- filter coefficients for filtering the beta-band from the signal	
	"""
	
	lfp_beta_signal = signal.filtfilt(beta_b, beta_a, lfp_signal)
	lfp_beta_signal_rectified = np.absolute(lfp_beta_signal)
	beta_ARV = np.mean(lfp_beta_signal_rectified[-2*tail_length:-tail_length])
	
	return beta_ARV
	
	
def calculate_tremor_ARV(force_signal, estimate_window_length, tail_length, tremor_b, tremor_a):
	""" Calculate the average rectified value of tremor-band activity in the simulated force signal window
	 
	  force_signal             - window of force signal
	  estimate_window_length - length of window to estimate the tremor biomarkers
	  tail_length           - length of tail that is discarded due to filtering artefact
	  tremor_b, tremor_a     - filter coefficients for lowpass filtering the force signal
	"""
	
	# Downsample the force signal from 1e5Hz to 100Hz
	force_signal = force_signal[0::1000]

	# Correspondingly reduce the input window lengths to accommodate the signal downsampling
	estimate_window_length = int(estimate_window_length/1e3)
	tail_length = int(tail_length/1e3)
	
	# Remove the mean from the full signal window
	force_signal = force_signal - np.mean(force_signal)
	
	# Bandpass filter the force signal at the tremor oscillation - 5 Hz
	filtered_force_signal = signal.filtfilt(tremor_b, tremor_a, force_signal)
	
	# Extract the portion of the signal required to calculate the biomarker
	force_biomarker_signal_segment = filtered_force_signal[-int(estimate_window_length+tail_length):-int(tail_length)]
	
	# Calculate the tremor biomarker - the ARV of the biomarker signal window	
	force_biomarker_signal_mean_removed = force_biomarker_signal_segment - np.mean(force_biomarker_signal_segment)
	tremor_ARV = np.mean(np.absolute(force_biomarker_signal_mean_removed))
	
	return tremor_ARV
	
def calculate_sEMG_ARV(sEMG_signal, estimate_window_length):
	""" Calculate the average rectified value of the EMG signal window to estimate muscle activation biomarker
	
	  sEMG_signal             - window of sEMG signal
	  estimate_window_length - length of window to estimate the muscle activation biomarker
	
	"""
	
	# Extract the portion of the signal required to calculate the biomarker
	sEMG_biomarker_signal_segment = sEMG_signal[-int(estimate_window_length)::]
	sEMG_biomarker_signal_segment = sEMG_biomarker_signal_segment[0::100]			# Downsample the signal from 1e5 Hz to 1e3 Hz
	
	# Calculate the sEMG ARV
	sEMG_ARV = np.mean(np.absolute(sEMG_biomarker_signal_segment))
	
	return sEMG_ARV

def generate_Beta_Period_Bias_Current(start_time, end_time, healthy_burst_duration, pathological_burst_duration_min, pathological_burst_duration_max, interburst_period):
	"""Generate current signal to simulate healthy and pathological beta bursts in CBG model. Cortical neurons 
		in the CBG model are biased in the main script to fire below the beta-band, so here 0 in the bias current corresponds to beta activity being off
		and 1 corresponds to beta activity being on
	
	Example inputs:
		start_time = 10100                               # ms
		end_time = 20000                                 # ms
		healthy_burst_duration = 100.0                   # ms - duration of a healthy burst of beta activity
		pathological_burst_duration_min = 600.0          # ms - minimum duration of a pahtological burst of beta activity
		pathological_burst_duration_max = 1000.0         # ms - maximum duration of a pahtological burst of beta activity
		interburst_period = 300.0                        # ms - time between the beta bursts
	Example outputs:
		beta_burst_times                       # An array of the beta burst onset times
		beta_burst_lvl                         # Defines whether the corresponding time is an on or off beta burst time
	"""
	
	beta_burst_times = []
	beta_burst_lvl = []
	t = start_time
	
	# Generate the beta burst bias current signal
	while t < end_time:
		
		# Choose from uniform random distribution to simulate physiological or 
		# pathological beta burst
		burst_type = np.random.uniform(0,1.0)
		
		# Check the burst type
		#if burst_type < 0.5:	# 50 % chance of healthy beta burst duration
		#if burst_type < 0.25:	# 25 % chance of healthy beta burst duration
		#if burst_type < 0.2:	# 20 % chance of healthy beta burst duration
		if burst_type < 0.0:	# 0 % chance of healthy beta burst duration - during elevated beta period all bursts are simulated as pathological
			# Healthy burst
			beta_burst_times = np.hstack((beta_burst_times, t, t+healthy_burst_duration))
			beta_burst_lvl = np.hstack((beta_burst_lvl, 1, 0))			# Positive phases correspond to beta on period
			t = t + healthy_burst_duration + interburst_period
		else:
			# Pathological burst
			pathological_burst_duration = np.round(np.random.uniform(pathological_burst_duration_min, pathological_burst_duration_max), 2)
			beta_burst_times = np.hstack((beta_burst_times, t, t+pathological_burst_duration))
			beta_burst_lvl = np.hstack((beta_burst_lvl, 1, 0))			# Positive phases correspond to beta on period
			t = t + pathological_burst_duration + interburst_period
	
	return beta_burst_times, beta_burst_lvl
	
def generate_Tremor_Period_Bias_Currents(pop_size, start_time, end_time, burst_duration, burst_frequency, onset_variability):
	"""Generate current signals to simulate tremor bursts in CBG model
	
	Example inputs:
		pop_size = 100                           # Size of the neuron population the currents are being generated for
		start_time = 40000                       # ms
		end_time = 70000                         # ms
		burst_duration = 50.0                    # mA - duration of the on period of the burst current
		burst_frequency = 5                      # Hz - How many burst occur per second
		onset_variability = 12.5                 # ms - variability of the onset times for the burst signals
	Example outputs:
		tremor_burst_times                       # A 2D array of the burst onset times for each neuron
		tremor_burst_lvl                         # Defines whether the corresponding time is an on or off burst time
	"""
	
	tremor_burst_times = []
	t = start_time
	interburst_period = np.ceil(((1.0/burst_frequency))*1e3)
	
	# Generate basic bias current signal with no variability
	while t < end_time:
		tremor_burst_times = np.hstack((tremor_burst_times, t, t+burst_duration))
		t = t + interburst_period
		
	tremor_burst_lvl = np.zeros(len(tremor_burst_times))
	tremor_burst_lvl[0::2]= 1
	
	# Make the burst onset times more variable for each neuron
	burst_onset_variability = np.round(np.random.uniform(0,onset_variability,(pop_size, len(tremor_burst_times))), 2)
	tremor_burst_times = tremor_burst_times + burst_onset_variability
	
	return tremor_burst_times, tremor_burst_lvl
	
def generate_Beta_and_Tremor_Period_Bias_Currents(pop_size, beta_start_time, beta_end_time, tremor_start_time, tremor_end_time, beta_current_max_amplitude,tremor_current_max_amplitude, healthy_beta_burst_duration, pathological_beta_burst_duration_min, pathological_beta_burst_duration_max, beta_interburst_period, 
													tremor_burst_duration, tremor_burst_frequency, tremor_burst_onset_variability):
	"""Generate current signals to simulate healthy and pathological beta activity and tremor activity in CBG model
	
	Example inputs:
		pop_size = 100                                        # Size of the neuron population the currents are being generated for
		beta_start_time = 10100                               # ms
		beta_end_time = 20000                                 # ms
		tremor_start_time = 20000                             # ms
		tremor_end_time = 30000                               # ms
		beta_current_max_amplitude = 0.076                    # nA - amplitude used to modulate the beta burst current
		tremor_current_max_amplitude = 90.0                   # nA - amplitude used to modulate the tremor burst current
		healthy_beta_burst_duration = 100.0                   # ms - duration of a healthy burst of beta activity
		pathological_beta_burst_duration_min = 600.0          # ms - minimum duration of a pahtological burst of beta activity
		pathological_beta_burst_duration_max = 1000.0         # ms - maximum duration of a pahtological burst of beta activity
		beta_interburst_period = 300.0                        # ms - time between the beta bursts
		tremor_burst_duration = 50.0                          # mA - duration of the on period of the burst current
		tremor_burst_frequency = 5                            # Hz - How many burst occur per second
		tremor_burst_onset_variability = 30.0                 # ms - variability of the onset times for the burst signals
	Example outputs:
		tremor_burst_times                                    # A 2D array of the burst onset times for each neuron
		tremor_burst_lvl                                      # Defines whether the corresponding time is an on or off burst time
		beta_burst_times                                      # An array of the beta burst onset times
		beta_burst_lvl                                        # Defines whether the corresponding time is an on or off beta burst time
	"""
	
	# Check to make sure tremor is not starting in the middle of the final beta burst
	if tremor_start_time < beta_end_time+pathological_beta_burst_duration_min:
		tremor_start_time = tremor_start_time + pathological_beta_burst_duration_min
		tremor_end_time = tremor_end_time + pathological_beta_burst_duration_min
		
	# Generate the beta burst signal
	beta_burst_times, beta_burst_lvl = generate_Beta_Period_Bias_Current(beta_start_time, beta_end_time, healthy_beta_burst_duration, pathological_beta_burst_duration_min, pathological_beta_burst_duration_max, beta_interburst_period)
	beta_burst_lvl = beta_current_max_amplitude * beta_burst_lvl
	
	# Generate the tremor burst signal
	tremor_burst_times, tremor_burst_lvl = generate_Tremor_Period_Bias_Currents(pop_size, tremor_start_time, tremor_end_time, tremor_burst_duration, tremor_burst_frequency, tremor_burst_onset_variability)
	tremor_burst_lvl = tremor_current_max_amplitude * tremor_burst_lvl
	
	# Concatenate the signals so they are consistent for loading into the cortical and thalamic neuron populations
	cortical_bias_current_signal_times = np.hstack((beta_burst_times, tremor_burst_times[1,:]))
	cortical_bias_current_signal_lvl = np.hstack((beta_burst_lvl, -beta_current_max_amplitude*np.ones(len(tremor_burst_times[1,:]))))
	thalamic_bias_current_signal_times = np.hstack((np.tile(beta_burst_times, (pop_size, 1)), tremor_burst_times))
	thalamic_bias_current_signal_lvl = np.hstack((np.zeros(len(beta_burst_lvl)), tremor_burst_lvl))
	
	return cortical_bias_current_signal_times, cortical_bias_current_signal_lvl, thalamic_bias_current_signal_times, thalamic_bias_current_signal_lvl

def calculate_Force_Twitch(spike_time, IPI, P, T, twitch_duration, dt):
	"""calculate_Force_Twitch Generates a single force twitch signal using the Fuglevand model 
		This function generates a single force twitch signal using the
		Fuglevand model for a spike at time spike_time with interpulse interval (IPI), 
		with twitch duration and twitch parameters P and T. Units of time are in ms
		for Fuglevand model.
	
		Calculate the sigmoidal gain value for the force model 
		with specified:
		T - twitch contraction time 
		IPI - instantaneous interpulse interval
	"""
	
	if T/IPI <= 0.4:
		g = 1.0
	else:
		normalization_factor = (1.0 - np.exp(-2.0*(0.4)**3))
		g = (1.0 - np.exp(-2.0*(T/IPI)**3))/(T/IPI)
		g = g/normalization_factor
	
	# Make time vector for the MU twitch signal
	t = np.arange(0, twitch_duration, dt)
	
	# Generate the twitch force signal
	force_twitch_signal = g*((P*t)/T)*np.exp(1.0-(t/T))
	
	# Update t so spike corresponds to correct time signal
	t = t + spike_time
	
	return t, force_twitch_signal

def calculate_Force_Twitch_Train(t, spike_times, IPI_values, P, T, twitch_duration, dt):
	"""calculate_Force_Twitch_Train Generates force signal from 
		a train of spike_times with interpulse interval values using the 
		Fuglevand force twitch model for force production.
		This function generates the force signal using the
		Fuglevand model for a train of impluses with interpulse interval (IPI) 
		where each spike occurs at a corresponding time in spike_time array, 
		with specified twitch duration and twitch contraction parameters are P 
		and T. Units of time are in ms for Fuglevand model.
	"""
	# Initialize place holder force signal
	force_train_signal = np.zeros(len(t)).flatten()
	
	# Check if there are actually any spikes
	if len(spike_times) >= 1 and len(IPI_values) >= 1:
		# Go through the spike times and calculate the corresponding force twitches 
		for (spike_time, IPI_value) in zip(spike_times, IPI_values):
		
			# Calculate the twitch force for each spike in the train 
			twitch_signal_t,force_twitch_signal = calculate_Force_Twitch(spike_time, IPI_value, P, T, twitch_duration, dt)
			
			# Make sure twitch signal is rounded to two decimal places
			twitch_signal_t = np.around(twitch_signal_t, 2)
			
			# Find the indices of the single force twitch times in the 
			# overall force signal for the train of impulses.
			start_index = np.where(t==twitch_signal_t[0])[0][0]
			end_index = np.where(t==twitch_signal_t[-1])[0]		# don't pull the exact end_index value for error checking
			
			# Check if the last twitch runs past the end of the time vector
			if end_index.size == 0:
				# Only add the relevant part of the twitch to the force signal
				end_index = np.where(twitch_signal_t==t[-1])[0][0]
				force_train_signal[start_index::] = force_train_signal[start_index::] + force_twitch_signal[0:end_index+1]
			else:
				# Not running over end of time vector, so can pull the end_index value now
				end_index = end_index[0]	
				# Add the single force twitch signal to the overall force train signal
				force_train_signal[start_index:end_index+1] = force_train_signal[start_index:end_index+1] + force_twitch_signal

	return t, force_train_signal

def calculate_Total_Pool_Force(t, total_pool_force_signal, pool_spike_trains, P_values, T_values, twitch_duration, dt, initial_spike_times):
	"""calculate_Total_Pool_Force Generates the force signal from 
		the motoneuron pool model. Peak twitch force, P_values, and twitch
		contraction time, T_values, are arrays of values distributed the
		parameters over the pool. pool_spike_trains are the corresponding 1s and 0s
		for spikes from the MU pool occuring with time t.
		Inputs:
			t = np.arange(0, 3000.0, dt)    						# time vector for overall force signal defined in main code
			total_pool_force_signal = np.zeros(len(t)).flatten()   	# placeholder total pool force signal initialize as zero vector in main script
			pool_spike_trains 										# ms - spike times pulled from the pool
			P_values 												# A.U. - peak twitch contraction force for each MU
			T_values 												# ms - peak twitch contraction time for each MU
			twitch_duration 										# ms - the duration of each force twitch
			dt 														# ms - sampling rate of the force signal
			initial_spike_times   									# ms - array of initial spike times to calculate initial IPI of each spike train with respect to
	"""
	
	# Initialize array to hold end spike times
	last_spike_times = initial_spike_times
	
	# Loop over the population and calculate the total force signal
	for i, spike_train in enumerate(pool_spike_trains):
		spike_times = np.around(spike_train.magnitude, 2)			# pool_spike_trains uses the quantities package, so spike time values need to be extracted first - also round to two decimal places
		if len(spike_times) >= 1:	# Check to make sure spikes happened, otherwise don't calculate force
			
			# calculate each IPI with respect to each initial spike time
			IPI_values = np.diff(np.hstack((initial_spike_times[i], spike_times))) 
			
			# Calculate the force signal due to the spike times
			t,force_train_signal = calculate_Force_Twitch_Train(t, spike_times, IPI_values, P_values[i], T_values[i], twitch_duration, dt)
			
			# Add the force signal to the total overall force signal for the MU pool
			total_pool_force_signal = total_pool_force_signal + force_train_signal
			
			# Update the last time a spike occurred for the respective MU
			last_spike_times[i] = spike_times[-1]	
				
	return t, total_pool_force_signal, last_spike_times

def generate_MU_Pool_Force_Twitch_Parameters(Pop_size, R_P, T_max):
	"""generate_MU_Pool_Force_Twitch_Parameters Generate force twitch parameters for neurons in motoneuron pool
		Example inputs:
			Pop_size = 100		% Neurons in MU Pool
			R_P = 100			
			T_max = 90			% ms 
			Outputs:
			P 	    	        % Array of Peak twitch forces for each MU in pool
			T 	    	        % Array of twitch contraction times for each MU in pool
	"""
	# Calculate peak twitch force parameters - P
	P = np.arange(0,Pop_size)
	b = np.log(R_P)/Pop_size
	P = np.exp(b*P)
	
	# Calculate twitch contraction times parameter - T
	T = T_max*(1.0/P)**(1/4.2)
	
	return P, T	
	
def calculate_sEMG_Twitch(spike_time, MU_Action_Potential_shape, MU_Action_Potential_shape_duration, dt):
	"""calculate_Force_Twitch Generates a single force twitch signal using the Fuglevand model 
		This function generates a single force twitch signal using the
		Fuglevand model for a spike at time spike_time with interpulse interval (IPI), 
		with twitch duration and twitch parameters P and T. Units of time are in ms
		for Fuglevand model.
	
		Calculate the sigmoidal gain value for the force model 
		with specified:
		T - twitch contraction time 
		IPI - instantaneous interpulse interval
	"""
	
	# Make time vector for the MU twitch signal
	t = np.arange(0, MU_Action_Potential_shape_duration, dt)
	
	# Generate placeholder sEMG twitch signal
	sEMG_twitch_signal = np.zeros(len(t))
	sEMG_twitch_signal[0] = 1
	
	# Generate the sEMG twitch signal
	sEMG_twitch_signal = np.convolve(sEMG_twitch_signal, MU_Action_Potential_shape)
	
	# Update the time vector - convolved signal will be longer than original shape
	t = np.around(dt * np.arange(0, len(sEMG_twitch_signal)), 2)
	
	# Update t so spike corresponds to correct time signal
	t = np.around(t + spike_time, 2)
	
	return t, sEMG_twitch_signal

def calculate_sEMG_Twitch_Train(t, spike_times, MU_Action_Potential_shape, sEMG_shape_duration, dt):
	"""calculate_Force_Twitch_Train Generates force signal from 
		a train of spike_times with interpulse interval values using the 
		Fuglevand force twitch model for force production.
		This function generates the force signal using the
		Fuglevand model for a train of impluses with interpulse interval (IPI) 
		where each spike occurs at a corresponding time in spike_time array, 
		with specified twitch duration and twitch contraction parameters are P 
		and T. Units of time are in ms for Fuglevand model.
	"""
	# Initialize place holder sEMG signal
	sEMG_train_signal = np.zeros(len(t)).flatten()
	
	# Check if there are actually any spikes
	if len(spike_times) >= 1:
		# Go through the spike times and calculate the corresponding force twitches 
		for spike_time in spike_times:
		
			# Calculate the twitch force for each spike in the train 
			sEMG_twitch_signal_t,sEMG_twitch_signal = calculate_sEMG_Twitch(spike_time, MU_Action_Potential_shape, sEMG_shape_duration, dt)
			
			# Make sure twitch signal is rounded to two decimal places
			sEMG_twitch_signal_t = np.around(sEMG_twitch_signal_t, 2)
			
			# Find the indices of the single force twitch times in the 
			# overall force signal for the train of impulses.
			start_index = np.where(t==sEMG_twitch_signal_t[0])[0][0]
			end_index = np.where(t==sEMG_twitch_signal_t[-1])[0]		# don't pull the exact end_index value for error checking
			
			# Check if the last twitch runs past the end of the time vector
			if end_index.size == 0:
				# Only add the relevant part of the twitch to the force signal
				end_index = np.where(sEMG_twitch_signal_t==t[-1])[0][0]
				sEMG_train_signal[start_index::] = sEMG_train_signal[start_index::] + sEMG_twitch_signal[0:end_index+1]
			else:
				# Not running over end of time vector, so can pull the end_index value now
				end_index = end_index[0]	
				# Add the single force twitch signal to the overall force train signal
				sEMG_train_signal[start_index:end_index+1] = sEMG_train_signal[start_index:end_index+1] + sEMG_twitch_signal

	return t, sEMG_train_signal

def calculate_MU_Pool_sEMG(t, total_pool_sEMG_signal, pool_spike_trains, MU_Action_Potential_shapes, sEMG_shape_duration, dt):
	"""calculate_Total_Pool_Force Generates the force signal from 
		the motoneuron pool model. Peak twitch force, P_values, and twitch
		contraction time, T_values, are arrays of values distributed the
		parameters over the pool. pool_spike_trains are the corresponding 1s and 0s
		for spikes from the MU pool occuring with time t.
		Inputs:
			t = np.arange(0, 3000.0, dt)    						# time vector for overall force signal defined in main code
			total_pool_force_signal = np.zeros(len(t)).flatten()   	# placeholder total pool force signal initialize as zero vector in main script
			pool_spike_trains 										# ms - spike times pulled from the pool
			P_values 												# A.U. - peak twitch contraction force for each MU
			T_values 												# ms - peak twitch contraction time for each MU
			twitch_duration 										# ms - the duration of each force twitch
			dt 														# ms - sampling rate of the force signal
			initial_spike_times   									# ms - array of initial spike times to calculate initial IPI of each spike train with respect to
	"""
	
	# Error code - make sure t vector is rounded to 2 decimal places
	t = np.around(t, 2)
	
	# Loop over the population and calculate the total sEMG signal
	for i, spike_train in enumerate(pool_spike_trains):

		spike_times = np.around(spike_train.magnitude, 2)			# pool_spike_trains uses the quantities package, so spike time values need to be extracted first - also round to two decimal places
		if len(spike_times) >= 1:	# Check to make sure spikes happened, otherwise don't calculate force
			
			# Calculate the force signal due to the spike times
			t,sEMG_train_signal = calculate_sEMG_Twitch_Train(t, spike_times, MU_Action_Potential_shapes[i], sEMG_shape_duration, dt)
			
			# Add the force signal to the total overall force signal for the MU pool
			total_pool_sEMG_signal = total_pool_sEMG_signal + sEMG_train_signal
			
	return t, total_pool_sEMG_signal
	
def generate_exponentially_distributed_parameters(rate_parameter, lower_bound, upper_bound, Pop_size):
	"""generate_exponentially_distributed_parameters Generates an exponentially distributed array of parameters. 
		Intended for use with motoneurons, so array is returned in reverse order, i.e. larger values are for earlier 
		recruited units.
		Example inputs:
			rate_parameter = 0.1 % The rate of decay of the distribution - a higher value makes the distribution decrease faster
			lower_bound = 0		% lower bound of the parameter
			upper_bound = 100	% upper bound of the parameter
			Pop_size 	= 100 	% Size of the population
			Outputs:
			parameter_values 	% Array of exponentially distributed values for the MU pool
	"""
	
	# Generate the exponential distribution
	x = np.linspace(0,Pop_size,Pop_size)
	parameter_values = rate_parameter*np.exp(-rate_parameter * x)
	
	# Rescale the distribution to be in correct parameter bounds
	parameter_values = lower_bound + (upper_bound - lower_bound)* parameter_values/rate_parameter
	
	return parameter_values
	
def generate_linearly_distributed_parameters(lower_bound, upper_bound, Pop_size, slope_start):
	"""Generate noise current for simulating noisy inputs to the dendrites
	
	Example inputs:
		lower_bound = 0 	# Lower parameter bound
		upper_bound = 1 	# Upper parameter bound
		Pop_size = 100  	# Number of parameters required
		slope_start = 0.4 	# Percentage of the population before which the parameter is constant at the lower bound
	"""
	
	parameter_values = lower_bound * np.ones(Pop_size)
	
	if slope_start <= 0:
		slope_start_index = 0
	elif slope_start > 1:
		slope_start_index = Pop_size - 1 
	else:
		slope_start_index = int(slope_start * Pop_size) - 1 
	
	# Generate linear ramp from lower to upper parameter bound values
	parameter_values[slope_start_index::] = np.linspace(lower_bound, upper_bound, len(parameter_values[slope_start_index::]))
	
	return parameter_values
	
if __name__ == '__main__':
	
	# Initialize DBS parameters, will be updated by controllers later on in main run loop
	DBS_amplitude_value = 0.0									
	#DBS_pulse_width_value = 0.06
	DBS_pulse_width_value = 60.0	# use us here, but convert to ms when building DBS signals later
	
	# Setup simulation
	setup(timestep=0.01, rngseed=3695)
	steady_state_duration = 8000.0														# Steady State simulation time
	
	# Define the motor symptom periods - 5s No Symtoms Period -> 10s Tremor Period -> 5s No Symptoms Period -> 10s Elevated Beta Period
	motor_symptom_state_durations = 30000.0																					# Duration of each motor symptom
	non_motor_performance_state_durations = 5000.0																			# Duration of normal conditions -  no elevated symptoms
	tremor_period_start_time = steady_state_duration + 2000.0 + non_motor_performance_state_durations						# Start time of tremor period
	tremor_period_end_time = tremor_period_start_time + motor_symptom_state_durations										# End time of tremor period
	#beta_period_start_time = tremor_period_end_time + non_motor_performance_state_durations									# Start time of beta period - 2 s window need to estimate the biomarkers so wait for this window to pass
	beta_period_start_time = tremor_period_end_time																			# Start time of beta period - 2 s window need to estimate the biomarkers so wait for this window to pass
	beta_period_end_time = beta_period_start_time + motor_symptom_state_durations											# End time of beta period
	
	# Total simulation duration
	simulation_duration = beta_period_end_time + 0.01										# Total simulation time
	
	# Recording sampling frequency for model populations and population size definition
	rec_sampling_interval = 0.5						# Fs = 2000Hz
	Pop_size = 100
	
	# Make a beta band filter centred on 26 Hz with 8 Hz bandwidth (cutoff frequency 22-30 Hz) for LFP beta ARV estimation
	filter_centre_frequency = 26			# Peak in LFP was centred on 25 Hz in original paper, now centred on 26 Hz 
	filter_bandwidth = 8
	beta_b, beta_a = make_beta_cheby1_filter(Fs=(1.0/rec_sampling_interval)*1000, N=4, rp=0.5, low=filter_centre_frequency-(filter_bandwidth/2.0), high=filter_centre_frequency+(filter_bandwidth/2.0))
	
	# Make a bandpass filter centred on 5 Hz with a 2 Hz bandwidth for force biomarker estimation - force is generated with 1e5 Hz sampling frequency - same as model timestep
	force_signal_Fs = 1e3*(1.0/10.0)			# Set filter sampling frequency to 100 Hz, will downsample force signal to 100 Hz in tremor biomarker calculation function
	tremor_filter_centre_frequency = 5			# Peak oscillation in the output force signal during tremor periods
	tremor_filter_bandwidth = 2
	tremor_b,tremor_a = signal.butter(4, [(tremor_filter_centre_frequency-(tremor_filter_bandwidth/2.0))/(force_signal_Fs/2.0), (tremor_filter_centre_frequency+(tremor_filter_bandwidth/2.0))/(force_signal_Fs/2.0)], btype='bandpass')
	
	# Use CVode to calculate i_membrane_ for fast LFP calculation
	cvode = h.CVode()
	cvode.active(0)
	
	# Get the second spatial derivative (the segment current) for the collateral
	cvode.use_fast_imem(1)
	
	# Set initial values for cell membrane voltages
	v_init = -68

	# Create random distribution for cell membrane noise current
	r_init = RandomDistribution('uniform',(0, Pop_size))
	
	# Create Spaces for STN Population
	STN_Electrode_space=space.Space(axes='xy')
	STN_space = space.RandomStructure(boundary=space.Sphere(2000))				# Sphere with radius 2000um
	
	"""
	# Generate poisson distributed spike time striatal input
	striatal_spike_times = generate_poisson_spike_times(pop_size=Pop_size, start_time=1.0, duration=simulation_duration, fr=25.0, timestep=1, random_seed=365)
	np.save('Striatal_Spike_Times.npy', striatal_spike_times)	
	"""
	
	# Load the striatal spike times
	striatal_spike_times =  np.load('Striatal_Spike_Times.npy')					# Load spike times from file
	
	"""
	# Generate 25 Hz poisson distributed spike times for independent inputs to the MU pool
	independent_beta_spike_times = generate_poisson_spike_times(pop_size=600, start_time=beta_period_start_time, duration=beta_period_end_time-beta_period_start_time, fr=25.0, timestep=1, random_seed=365)
	np.save('Independent_Beta_Spike_Times.npy', independent_beta_spike_times)	
	"""
	
	# Load the independent inputs spike times
	independent_beta_spike_times = np.load('Independent_Beta_Spike_Times.npy')					# Load spike times from file
	
	# Generate the cortico-basal ganglia neuron populations
	Cortical_Pop = Population(Pop_size, Cortical_Neuron_Type(soma_bias_current_amp=0.24), structure=STN_space, label='Cortical Neurons') 		# Neurons will fire at 26 hz for this bias current amplitude
	Interneuron_Pop = Population(Pop_size, Interneuron_Type(bias_current_amp=0.070), initial_values={'v': v_init}, label='Interneurons')
	STN_Pop = Population(Pop_size, STN_Neuron_Type(bias_current=-0.125), structure=STN_space, initial_values={'v': v_init}, label='STN Neurons')
	GPe_Pop = Population(Pop_size, GP_Neuron_Type(bias_current=-0.009), initial_values={'v': v_init}, label='GPe Neurons')					# GPe/i have the same parameters, but different bias currents 
	GPi_Pop = Population(Pop_size, GP_Neuron_Type(bias_current=0.006), initial_values={'v': v_init}, label='GPi Neurons')					# GPe/i have the same parameters, but different bias currents 
	Striatal_Pop = Population(Pop_size, SpikeSourceArray(spike_times=striatal_spike_times[0][0]), label='Striatal Neuron Spike Source')	
	Thalamic_Pop = Population(Pop_size, Thalamic_Neuron_Type(), initial_values={'v': v_init}, label='Thalamic Neurons')
	
	# Generate the pool parameters
	motoneuron_pool_parameters = generate_motoneuron_population_parameters(Pop_size, motoneuron_pool_parameter_bounds)
	
	# Generate motoneuron pool and independent input populations
	Motoneuron_Pop = Population(Pop_size, Spinal_Motoneuron_Synaptic_Bombardment_Point_Process_Type(**motoneuron_pool_parameters), initial_values={'v': v_init}, label='Motoneurons')
	Independent_Motoneuron_Input_Pop = Population(6*Pop_size, SpikeSourceArray(spike_times=independent_beta_spike_times[0][0]), label='Motoneuron Pool Independent Spike Source Population')	
	
	# Update the spike times for the independent beta source for the MU pool
	for i in range(0,6*Pop_size):
		Independent_Motoneuron_Input_Pop[i].spike_times=independent_beta_spike_times[i][0]
	
	# Update the spike times for the independent beta source for the MU pool
	for i in range(0,6*Pop_size):
		Independent_Motoneuron_Input_Pop[i].spike_times=independent_beta_spike_times[i][0]
	
	################################################################################################################################################################
	# Modulation Currents for the cortical and thalamic populations
	# Inject a baseline cortical modulation current to reduce the population firing rate
	baseline_cortical_bias_current = DCSource(amplitude=-0.076, start=10000.0, stop=100000.0)
	Cortical_Pop.inject(baseline_cortical_bias_current)
	
	"""
	# Generate new Cortical Beta Bias Current
	beta_current_max_amplitude = 0.076
	#beta_burst_times, beta_burst_lvl = generate_Beta_Period_Bias_Current(start_time=beta_period_start_time, end_time=beta_period_end_time, healthy_burst_duration=100.0, pathological_burst_duration_min=600.0, pathological_burst_duration_max=1000.0, interburst_period=300.0)
	beta_burst_times, beta_burst_lvl = generate_Beta_Period_Bias_Current(start_time=beta_period_start_time, end_time=beta_period_end_time, healthy_burst_duration=100.0, pathological_burst_duration_min=600.0, pathological_burst_duration_max=1000.0, interburst_period=150.0)
	beta_burst_lvl = beta_current_max_amplitude * beta_burst_lvl
	
	# Save the beta period bias currents
	np.save('Cortical_Bias_Signal_Times.npy', beta_burst_times)
	np.save('Cortical_Bias_Current_Signal.npy', beta_burst_lvl)
	"""
	
	# Load the beta and tremor period bias currents
	beta_burst_times = np.load('Cortical_Bias_Signal_Times.npy')
	beta_burst_lvl = np.load('Cortical_Bias_Current_Signal.npy')

	# Inject the cortical modulation current into the cortical neuron
	cortical_beta_modulation_current = StepCurrentSource(times=beta_burst_times, amplitudes=beta_burst_lvl)
	Cortical_Pop.inject(cortical_beta_modulation_current)
	
	"""
	# Generate new Thalamic Tremor Bias Current
	tremor_current_max_amplitude = 90.0
	tremor_burst_times, tremor_burst_lvl = generate_Tremor_Period_Bias_Currents(pop_size=Pop_size, start_time=tremor_period_start_time, end_time=tremor_period_end_time, burst_duration=50.0, burst_frequency=5.0, onset_variability=30.0)
	tremor_burst_lvl = tremor_current_max_amplitude * tremor_burst_lvl
	
	# Save the beta and tremor period bias currents
	np.save('Thalamic_Bias_Signal_Times.npy', tremor_burst_times)
	np.save('Thalamic_Bias_Current_Signal.npy', tremor_burst_lvl)
	"""
	
	# Load the tremor period bias currents
	tremor_burst_times = np.load('Thalamic_Bias_Signal_Times.npy')
	tremor_burst_lvl = np.load('Thalamic_Bias_Current_Signal.npy')
	
	# Inject the thalamic modulation current into the thalamic neurons
	thalamic_tremor_modulation_currents = [StepCurrentSource(times=tremor_burst_times[count,:], amplitudes=tremor_burst_lvl) for count in range(Pop_size)]
	
	# Now inject the tremor currents into the thalamic neurons
	for Thalamic_Neuron, Thalamic_Neuron_tremor_current in zip(Thalamic_Pop, thalamic_tremor_modulation_currents):
		Thalamic_Neuron.inject(Thalamic_Neuron_tremor_current)
	
	# Generate Noisy current sources for cortical pyramidal and interneuron populations
	Cortical_Pop_Membrane_Noise = [NoisyCurrentSource(mean=0,stdev=0.005,start=0.0, stop=simulation_duration, dt=1.0) for count in range(Pop_size)]
	Interneuron_Pop_Membrane_Noise = [NoisyCurrentSource(mean=0,stdev=0.005,start=0.0, stop=simulation_duration, dt=1.0) for count in range(Pop_size)]
	
	# Inject each membrane noise current into each cortical and interneuron in network
	for Cortical_Neuron, Cortical_Neuron_Membrane_Noise in zip(Cortical_Pop, Cortical_Pop_Membrane_Noise):
		Cortical_Neuron.inject(Cortical_Neuron_Membrane_Noise)
	
	for Interneuron, Interneuron_Membrane_Noise in zip(Interneuron_Pop, Interneuron_Pop_Membrane_Noise):
		Interneuron.inject(Interneuron_Membrane_Noise)
	
	# Update the spike times for the striatal populations
	for i in range(0,Pop_size):
		Striatal_Pop[i].spike_times=striatal_spike_times[i][0]
	
	# Load cortical positions - Comment/Remove to generate new positions
	Cortical_Neuron_xy_Positions = np.loadtxt('cortical_xy_pos.txt', delimiter=',')
	Cortical_Neuron_x_Positions = Cortical_Neuron_xy_Positions[0,:]
	Cortical_Neuron_y_Positions = Cortical_Neuron_xy_Positions[1,:]
	
	# Set cortical xy positions to those loaded in
	for cell_id, Cortical_cell in enumerate(Cortical_Pop):
		Cortical_cell.position[0] = Cortical_Neuron_x_Positions[cell_id]
		Cortical_cell.position[1] = Cortical_Neuron_y_Positions[cell_id]
	
	# Load STN positions - Comment/Remove to generate new positions
	STN_Neuron_xy_Positions = np.loadtxt('STN_xy_pos.txt', delimiter=',')
	STN_Neuron_x_Positions = STN_Neuron_xy_Positions[0,:]
	STN_Neuron_y_Positions = STN_Neuron_xy_Positions[1,:]
	
	# Set STN xy positions to those loaded in
	for cell_id, STN_cell in enumerate(STN_Pop):
		STN_cell.position[0] = STN_Neuron_x_Positions[cell_id]
		STN_cell.position[1] = STN_Neuron_y_Positions[cell_id]
		STN_cell.position[2] = 500
	
	"""
	# Position Check - 	
	# 1) Make sure cells are bounded in 4mm space in x, y coordinates
	# 2) Make sure no cells are placed inside the stimulating/recording electrode -0.5mm<x<0.5mm, -1.5mm<y<2mm
	for Cortical_cell in Cortical_Pop:
		#while(((np.abs(Cortical_cell.position[0])>2000) or ((np.abs(Cortical_cell.position[1])>2000))) or ((np.abs(Cortical_cell.position[0])<500) and (Cortical_cell.position[1]>-1500 and Cortical_cell.position[1]<2000))):
		while(((np.abs(Cortical_cell.position[0])>3000) or ((np.abs(Cortical_cell.position[1])>3000))) or ((np.abs(Cortical_cell.position[0])<700) and Cortical_cell.position[1]>-1500)):
			Cortical_cell.position = STN_space.generate_positions(1).flatten()
	
	#np.savetxt('cortical_xy_pos.txt', Cortical_Pop.positions, delimiter=',')	# Save the generated cortical xy positions to a textfile
	
	for STN_cell in STN_Pop:
		#while(((np.abs(STN_cell.position[0])>2000) or ((np.abs(STN_cell.position[1])>2000))) or ((np.abs(STN_cell.position[0])<500) and (STN_cell.position[1]>-1500 and STN_cell.position[1]<2000))):
		while(((np.abs(STN_cell.position[0])>2000) or ((np.abs(STN_cell.position[1])>2000))) or ((np.abs(STN_cell.position[0])<635) and STN_cell.position[1]>-1500)):
			STN_cell.position = STN_space.generate_positions(1).flatten()
	
	#np.savetxt('STN_xy_pos.txt', STN_Pop.positions, delimiter=',')	# Save the generated STN xy positions to a textfile
	"""
	
	# Assign Positions for recording and stimulating electrode point sources
	recording_electrode_1_position = np.array([0,-1500,250])
	recording_electrode_2_position = np.array([0,1500,250])
	stimulating_electrode_position = np.array([0,0,250])
	
	# Calculate STN cell distances to each recording electrode - only using xy coordinates for distance calculations
	STN_recording_electrode_1_distances = distances_to_electrode(recording_electrode_1_position, STN_Pop)
	STN_recording_electrode_2_distances = distances_to_electrode(recording_electrode_2_position, STN_Pop)
	
	# Calculate Cortical Collateral distances from the stimulating electrode - uses xyz coordinates for distance calculation - these distances need to be in um for xtra
	Cortical_Collateral_stimulating_electrode_distances = collateral_distances_to_electrode(stimulating_electrode_position, Cortical_Pop, L=500, nseg=11)
	#np.savetxt('cortical_collateral_electrode_distances.txt', Cortical_Collateral_stimulating_electrode_distances, delimiter=',')	# Save the generated cortical collateral stimulation electrode distances to a textfile
	
	# Synaptic Connections
	# Add variability to Cortical connections - cortical interneuron connection weights are random from uniform distribution
	gCtxInt_max_weight = 2.5e-3									# Ctx -> Int max coupling value
	gIntCtx_max_weight = 6.0e-3									# Int -> Ctx max coupling value
	gCtxInt = RandomDistribution('uniform',(0, gCtxInt_max_weight), rng=NumpyRNG(seed=3695))
	gIntCtx = RandomDistribution('uniform',(0, gIntCtx_max_weight), rng=NumpyRNG(seed=3695))
	
	# Define other synaptic connection weights and delays
	syn_CorticalAxon_Interneuron = StaticSynapse(weight=gCtxInt, delay=2)	
	syn_Interneuron_CorticalSoma = StaticSynapse(weight=gIntCtx, delay=2)
	syn_CorticalSpikeSourceCorticalAxon = StaticSynapse(weight=0.25, delay=0)
	syn_CorticalCollateralSTN = StaticSynapse(weight=0.12, delay=1)
	syn_STNGPe = StaticSynapse(weight=0.111111, delay=4)
	syn_GPeGPe = StaticSynapse(weight=0.015, delay=4)
	syn_GPeSTN = StaticSynapse(weight=0.111111, delay=3)
	syn_StriatalGPe = StaticSynapse(weight=0.01, delay=1)
	syn_STNGPi = StaticSynapse(weight=0.111111, delay=2)
	syn_GPeGPi = StaticSynapse(weight=0.111111, delay=2)
	syn_GPiThalamic = StaticSynapse(weight=3.0, delay=2)
	syn_ThalamicCortical = StaticSynapse(weight=0.00899, delay=2)
	syn_CorticalThalamic = StaticSynapse(weight=0.0, delay=2)
	
	# Load network topology from file
	# Cortical Basal Ganglia Network Connections:
	prj_CorticalAxon_Interneuron = Projection(Cortical_Pop, Interneuron_Pop,  FromFileConnector("CorticalAxonInterneuron_Connections.txt"), syn_CorticalAxon_Interneuron, source='middle_axon_node', receptor_type='AMPA')
	prj_Interneuron_CorticalSoma = Projection(Interneuron_Pop, Cortical_Pop,  FromFileConnector("InterneuronCortical_Connections.txt"), syn_Interneuron_CorticalSoma, receptor_type='GABAa')
	prj_CorticalSTN = Projection(Cortical_Pop, STN_Pop, FromFileConnector("CorticalSTN_Connections.txt"), syn_CorticalCollateralSTN, source='collateral(0.5)', receptor_type='AMPA')
	prj_STNGPe = Projection(STN_Pop, GPe_Pop, FromFileConnector("STNGPe_Connections.txt"), syn_STNGPe, source='soma(0.5)', receptor_type='AMPA')
	prj_GPeGPe = Projection(GPe_Pop, GPe_Pop, FromFileConnector("GPeGPe_Connections.txt"), syn_GPeGPe, source='soma(0.5)', receptor_type='GABAa')
	prj_GPeSTN = Projection(GPe_Pop, STN_Pop, FromFileConnector("GPeSTN_Connections.txt"), syn_GPeSTN, source='soma(0.5)', receptor_type='GABAa')	
	prj_StriatalGPe = Projection(Striatal_Pop, GPe_Pop, FromFileConnector("StriatalGPe_Connections.txt"), syn_StriatalGPe, source='soma(0.5)', receptor_type='GABAa')	
	prj_STNGPi = Projection(STN_Pop, GPi_Pop, FromFileConnector("STNGPi_Connections.txt"), syn_STNGPi, source='soma(0.5)', receptor_type='AMPA')
	prj_GPeGPi = Projection(GPe_Pop, GPi_Pop, FromFileConnector("GPeGPi_Connections.txt"), syn_GPeGPi, source='soma(0.5)', receptor_type='GABAa')
	prj_GPiThalamic = Projection(GPi_Pop, Thalamic_Pop, FromFileConnector("GPiThalamic_Connections.txt"), syn_GPiThalamic, source='soma(0.5)', receptor_type='GABAa')
	prj_ThalamicCortical = Projection(Thalamic_Pop, Cortical_Pop, FromFileConnector("ThalamicCorticalSoma_Connections.txt"), syn_ThalamicCortical, source='soma(0.5)', receptor_type='AMPA')
	prj_CorticalThalamic = Projection(Cortical_Pop, Thalamic_Pop, FromFileConnector("CorticalSomaThalamic_Connections.txt"), syn_CorticalThalamic, source='soma(0.5)', receptor_type='AMPA')
	
	# Need to get the Cortical->Thalamic synaptic weights, these are incorrectly updated post SS load, so need to reset to these values post SS reload
	g_Thalamic_Cortical_weights = prj_ThalamicCortical.get("weight",format="array")
	
	"""
	# Create new Synaptic Projections
	prj_CorticalAxon_Interneuron = Projection(Cortical_Pop, Interneuron_Pop,  FixedNumberPreConnector(n=10, allow_self_connections=False), syn_CorticalAxon_Interneuron, source='middle_axon_node', receptor_type='AMPA')
	prj_Interneuron_CorticalSoma = Projection(Interneuron_Pop, Cortical_Pop,  FixedNumberPreConnector(n=10, allow_self_connections=False), syn_Interneuron_CorticalSoma, receptor_type='GABAa')
	prj_CorticalSTN = Projection(Cortical_Pop, STN_Pop, FixedNumberPreConnector(n=5, allow_self_connections=False), syn_CorticalCollateralSTN, source='collateral(0.5)', receptor_type='AMPA')
	prj_STNGPe = Projection(STN_Pop, GPe_Pop, FixedNumberPreConnector(n=1, allow_self_connections=False), syn_STNGPe, source='soma(0.5)', receptor_type='AMPA')
	prj_GPeGPe = Projection(GPe_Pop, GPe_Pop, FixedNumberPreConnector(n=1, allow_self_connections=False), syn_GPeGPe, source='soma(0.5)', receptor_type='GABAa')
	prj_GPeSTN = Projection(GPe_Pop, STN_Pop, FixedNumberPreConnector(n=2, allow_self_connections=False), syn_GPeSTN, source='soma(0.5)', receptor_type='GABAa')	
	prj_StriatalGPe = Projection(Striatal_Pop, GPe_Pop, FixedNumberPreConnector(n=1, allow_self_connections=False), syn_StriatalGPe, source='soma(0.5)', receptor_type='GABAa')	
	prj_STNGPi = Projection(STN_Pop, GPi_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False), syn_STNGPi, source='soma(0.5)', receptor_type='AMPA')
	prj_GPeGPi = Projection(GPe_Pop, GPi_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False), syn_GPeGPi, source='soma(0.5)', receptor_type='GABAa')
	prj_GPiThalamic = Projection(GPi_Pop, Thalamic_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False), syn_GPiThalamic, source='soma(0.5)', receptor_type='GABAa')
	prj_ThalamicCortical = Projection(Thalamic_Pop, Cortical_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False), syn_ThalamicCortical, source='soma(0.5)', receptor_type='AMPA')
	prj_CorticalThalamic = Projection(Cortical_Pop, Thalamic_Pop, FixedNumberPreConnector(n=1,allow_self_connections=False), syn_CorticalThalamic, source='soma(0.5)', receptor_type='AMPA')
	
	# Create new thalamic -> Cortical projection - want to induce bursts at the tremor frequency in cortical neurons due to thalamic input
	# Projection Parameters:
	g_thalamic_cortical_weight = 0.00899
	syn_ThalamicCortical = StaticSynapse(weight=g_thalamic_cortical_weight, delay=2)
	num_thalamic_cortical_connections = 6
	prj_ThalamicCortical = Projection(Thalamic_Pop, Cortical_Pop, FixedNumberPreConnector(n=num_thalamic_cortical_connections,allow_self_connections=False), syn_ThalamicCortical, source='soma(0.5)', receptor_type='AMPA')
	"""
	
	"""
	# Save the network topology so it can be reloaded 
	#prj_CorticalSpikeSourceCorticalSoma.saveConnections(file="CorticalSpikeSourceCorticalSoma_Connections.txt")
	prj_CorticalAxon_Interneuron.saveConnections(file="CorticalAxonInterneuron_Connections.txt")
	prj_Interneuron_CorticalSoma.saveConnections(file="InterneuronCortical_Connections.txt")
	prj_CorticalSTN.saveConnections(file="CorticalSTN_Connections.txt")
	prj_STNGPe.saveConnections(file="STNGPe_Connections.txt")
	prj_GPeGPe.saveConnections(file="GPeGPe_Connections.txt")
	prj_GPeSTN.saveConnections(file="GPeSTN_Connections.txt")
	prj_StriatalGPe.saveConnections(file="StriatalGPe_Connections.txt")
	prj_STNGPi.saveConnections(file="STNGPi_Connections.txt")
	prj_GPeGPi.saveConnections(file="GPeGPi_Connections.txt")
	prj_GPiThalamic.saveConnections(file="GPiThalamic_Connections.txt")
	prj_ThalamicCortical.saveConnections(file="ThalamicCorticalSoma_Connections.txt")
	prj_CorticalThalamic.saveConnections(file="CorticalSomaThalamic_Connections.txt")
	"""
	
	# Cortical Motoneuron Pool Connections:
	# Define MU correlated input synaptic parameters
	g_CorticalAxon_Motoneuron = np.logspace(-5, -1, 31)[25]			# take the 26th value from this array - value identified from parameter sweep
	num_Correlated_Inputs = 25
	
	# Motoneuron Synapse definitions:
	# Depressing Synapses:
	syn_CorticalAxon_Motoneuron = TsodyksMarkramSynapse(U=0.3, tau_rec=130, tau_facil=17, weight=g_CorticalAxon_Motoneuron, delay=9.8)		# Depressing Synapses - filter out high frequency cortical inputs during high frequency DBS
	
	"""
	# Load CorticalAxon to MU pool topology from file
	prj_CorticalAxon_Interneuron = Projection(Cortical_Pop, Motoneuron_Pop,  FromFileConnector("CorticalAxon_Motoneuron_Connections_1.txt"), syn_CorticalAxon_Motoneuron, source='end_axon_node', receptor_type='d1_correlated_exc')
	"""
	
	# Motoneuron Projection Definitions:
	# Dendritic Projections - 
	# Correlated inputs - 
	prj_CorticalAxon_Motoneuron_1 = Projection(Cortical_Pop, Motoneuron_Pop, FixedNumberPreConnector(num_Correlated_Inputs, allow_self_connections=False, with_replacement=False), syn_CorticalAxon_Motoneuron, source='end_axon_node', receptor_type='d1_correlated_exc')
	prj_CorticalAxon_Motoneuron_2 = Projection(Cortical_Pop, Motoneuron_Pop, CloneConnector(prj_CorticalAxon_Motoneuron_1), syn_CorticalAxon_Motoneuron, source='end_axon_node', receptor_type='d2_correlated_exc')
	prj_CorticalAxon_Motoneuron_3 = Projection(Cortical_Pop, Motoneuron_Pop, CloneConnector(prj_CorticalAxon_Motoneuron_1), syn_CorticalAxon_Motoneuron, source='end_axon_node', receptor_type='d3_correlated_exc')
	prj_CorticalAxon_Motoneuron_4 = Projection(Cortical_Pop, Motoneuron_Pop, CloneConnector(prj_CorticalAxon_Motoneuron_1), syn_CorticalAxon_Motoneuron, source='end_axon_node', receptor_type='d4_correlated_exc')
	
	# Save the connections before distribute the values exponentially
	prj_CorticalAxon_Motoneuron_1.saveConnections(file="CorticalAxon_Motoneuron_Connections_1.txt")
	
	# Generate updated gAxonMU synaptic weights so they are exponentially distributed across the pool, i.e. strongest connections to the earlier recruited units
	exp_distributed_synaptic_weight_values = generate_exponentially_distributed_parameters(rate_parameter=0.1, lower_bound=0.0, upper_bound=g_CorticalAxon_Motoneuron, Pop_size=Pop_size)
	
	# Get the array of synaptic strengths between the cortical axon and the MUs
	# Dendritic Projections
	g_CorticalAxon_Motoneuron_weights = prj_CorticalAxon_Motoneuron_1.get("weight",format="array")
	# Somatic Projections 
	#g_CorticalAxon_Motoneuron_weights = prj_CorticalAxon_Motoneuron.get("weight",format="array")
	
	# Now update the array weights to be exponentially distributed
	for idx,row in enumerate(g_CorticalAxon_Motoneuron_weights):
		update_indices = ~np.isnan(row)
		row[update_indices] = exp_distributed_synaptic_weight_values[update_indices]
	
	# Update the gAxonMU connection weights
	# Dendritic Projections - 
	prj_CorticalAxon_Motoneuron_1.set(weight=g_CorticalAxon_Motoneuron_weights)
	prj_CorticalAxon_Motoneuron_2.set(weight=g_CorticalAxon_Motoneuron_weights)
	prj_CorticalAxon_Motoneuron_3.set(weight=g_CorticalAxon_Motoneuron_weights)
	prj_CorticalAxon_Motoneuron_4.set(weight=g_CorticalAxon_Motoneuron_weights)
	
	##############################################################################
	# Independent Spike Source to MU pool connections
	g_Independent_Motoneuron = 0.02
	num_Independent_Inputs = 100
	
	# Motoneuron Synapse definitions:
	syn_Independent_Motoneuron = StaticSynapse(weight=g_Independent_Motoneuron, delay=9.8)		# Ref for delay : Lemon, R.N., Mantel, G.W. and Muir, R.B., 1986. Corticospinal facilitation of hand muscles during voluntary movement in the conscious monkey. The Journal of physiology, 381(1), pp.497-527.
	
	"""
	# Load Independent beta spike source inputs to MU pool topology from file
	prj_Independent_Motoneuron_1 = Projection(Independent_Motoneuron_Input_Pop, Motoneuron_Pop,  FromFileConnector("Independent_Motoneuron_Connections_1.txt"), syn_Independent_Motoneuron, source='soma(0.5)', receptor_type='d1_independent_exc')
	"""
	
	# Motoneuron Projection Definitions:
	# Dendritic Projections - 
	# Independent inputs - 
	prj_Independent_Motoneuron_1 = Projection(Independent_Motoneuron_Input_Pop, Motoneuron_Pop, FixedNumberPreConnector(num_Independent_Inputs, allow_self_connections=False, with_replacement=False), syn_Independent_Motoneuron, source='soma(0.5)', receptor_type='d1_independent_exc')
	prj_Independent_Motoneuron_2 = Projection(Independent_Motoneuron_Input_Pop, Motoneuron_Pop, CloneConnector(prj_Independent_Motoneuron_1), syn_Independent_Motoneuron, source='soma(0.5)', receptor_type='d2_independent_exc')
	prj_Independent_Motoneuron_3 = Projection(Independent_Motoneuron_Input_Pop, Motoneuron_Pop, CloneConnector(prj_Independent_Motoneuron_1), syn_Independent_Motoneuron, source='soma(0.5)', receptor_type='d3_independent_exc')
	prj_Independent_Motoneuron_4 = Projection(Independent_Motoneuron_Input_Pop, Motoneuron_Pop, CloneConnector(prj_Independent_Motoneuron_1), syn_Independent_Motoneuron, source='soma(0.5)', receptor_type='d4_independent_exc')
	
	# Generate exponential distribution of the synaptic weights
	#exp_distributed_independent_input_synaptic_weight_values = generate_exponentially_distributed_parameters(rate_parameter=0.05, lower_bound=0.0, upper_bound=g_Independent_Motoneuron, Pop_size=Pop_size)
	exp_distributed_independent_input_synaptic_weight_values = generate_exponentially_distributed_parameters(rate_parameter=0.035, lower_bound=0.0, upper_bound=g_Independent_Motoneuron, Pop_size=Pop_size)
	
	# Get the array of synaptic strengths between the independent inputs and the MU pool
	g_Independent_Motoneuron_weights = prj_Independent_Motoneuron_1.get("weight",format="array")
	
	# Now update the array weights to be exponentially distributed
	for idx,row in enumerate(g_Independent_Motoneuron_weights):
		update_indices = ~np.isnan(row)
		row[update_indices] = exp_distributed_independent_input_synaptic_weight_values[update_indices]
	
	# Update the gIndependent Input MU connection weights
	prj_Independent_Motoneuron_1.set(weight=g_Independent_Motoneuron_weights)
	prj_Independent_Motoneuron_2.set(weight=g_Independent_Motoneuron_weights)
	prj_Independent_Motoneuron_3.set(weight=g_Independent_Motoneuron_weights)
	prj_Independent_Motoneuron_4.set(weight=g_Independent_Motoneuron_weights)
	
	# Recorded variables from the cortical basal ganglia model
	Cortical_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
	Cortical_Pop.record('end_node(0.5).v', sampling_interval=rec_sampling_interval)
	Cortical_Pop.record('collateral(0.5).v', sampling_interval=rec_sampling_interval)
	Cortical_Pop.record('AMPA.i', sampling_interval=rec_sampling_interval)
	Cortical_Pop.record('GABAa.i', sampling_interval=rec_sampling_interval)
	Interneuron_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
	STN_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
	STN_Pop.record('AMPA.i', sampling_interval=rec_sampling_interval)
	STN_Pop.record('GABAa.i', sampling_interval=rec_sampling_interval)
	Striatal_Pop.record('spikes')
	GPe_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
	GPi_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
	Thalamic_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
	Thalamic_Pop.record('AMPA.i', sampling_interval=rec_sampling_interval)
	Thalamic_Pop.record('GABAa.i', sampling_interval=rec_sampling_interval)
	
	# Recorded variables from the motoneuron pool model
	Motoneuron_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
	#Motoneuron_Pop.record('soma_correlated_exc.i', sampling_interval=rec_sampling_interval)
	
	Motoneuron_Pop.record('d1_correlated_exc.i', sampling_interval=rec_sampling_interval)
	Motoneuron_Pop.record('d2_correlated_exc.i', sampling_interval=rec_sampling_interval)
	Motoneuron_Pop.record('d3_correlated_exc.i', sampling_interval=rec_sampling_interval)
	Motoneuron_Pop.record('d4_correlated_exc.i', sampling_interval=rec_sampling_interval)
	Motoneuron_Pop.record('spikes')
	
	# Record independent input spikes
	Independent_Motoneuron_Input_Pop.record('spikes')
	
	# Conductivity and resistivity values for homogenous, isotropic medium
	sigma = 0.27							# Latikka et al. 2001 - Conductivity of Brain tissue S/m
	rho = (1/(sigma*1e-2))					# rho needs units of ohm cm for xtra mechanism (S/m -> S/cm)
	
	"""
	# Load the cortical collateral resistivity values from file
	collateral_rx = np.loadtxt('cortical_collateral_resistivities.txt', delimiter=',')
	"""
	
	# Calculate transfer resistances for each collateral segment for xtra - units are Mohms
	collateral_rx = (rho/(4*math.pi))*(1/Cortical_Collateral_stimulating_electrode_distances)*(0.01)
	
	"""
	# Save the transfer resistances for each collateral segment to a textfile
	np.savetxt('cortical_collateral_rx.txt', collateral_rx, delimiter=',')	# Save the resistivity values between each cortical collateral segment and the stimulation electrode to a textfile
	"""
	
	# Convert ndarray to array of Sequence objects - needed to set cortical collateral transfer resistances
	collateral_rx_seq = np.ndarray(shape=(1, Pop_size), dtype=Sequence).flatten()
	for ii in range(0, Pop_size):
		collateral_rx_seq[ii] = Sequence(collateral_rx[ii,:].flatten())
	
	# Assign transfer resistances values to collaterals
	for ii, cortical_neuron in enumerate(Cortical_Pop):
		cortical_neuron.collateral_rx = collateral_rx_seq[ii]
	
	# Make ramped bias current for each dendrite on each motoneuron - each MU neuron needs pointers to it's dendritic bias current signals
	MU_Dendritic_Bias_Current_neuron = []
	MU_Dendritic_Bias_times_neuron = []
	
	# Ramped Noise current parameters - constant level of excitation
	mean_ramp_noise_value = 3.0
	stdev_ramp_noise_value = 0.75
	
	# Constant Noise current parameters - varying level of inhibition across the MU pool - more inhibition on bigger fibres
	mean_inhibtion_noise_value = -12
	stdev_inhibtion_noise_value = 0.75
	
	# Exponential increase in inhbibitory bias current for neurons above 40th MU 
	tremor_bursting_end_index = 40		# The MU index above which we want to suppress firing of the motoneurons in the pool, so only 40 % of pool is recruited during tremor periods
	mean_inhibition_noise_values = generate_exponentially_distributed_parameters(rate_parameter=0.1, lower_bound=mean_inhibtion_noise_value, upper_bound=0.0, Pop_size=int(Pop_size-tremor_bursting_end_index))
	stdev_inhibition_noise_values = generate_exponentially_distributed_parameters(rate_parameter=0.1, lower_bound=stdev_inhibtion_noise_value, upper_bound=0.0, Pop_size=int(Pop_size-tremor_bursting_end_index))
	mean_inhibition_noise_values = np.hstack((np.zeros(tremor_bursting_end_index),mean_inhibition_noise_values))
	stdev_inhibition_noise_values = np.hstack((np.zeros(tremor_bursting_end_index),stdev_inhibition_noise_values))
	
	# Loop over the MUs in the pool and put ramped dendritic noise current into each
	for i in range(0, Pop_size):
		
		# Generate a ramped excitatory noisy current for each MU - Noise is filtered at 150 Hz due to lowpass filtering properties of neuron membrane
		ramped_bias_current_times, ramped_noise_current_amp = generate_Ramped_Noise_Signal(start_time=0.0, stop_time=simulation_duration+10, dt=1.0, ramp_duration=6000.0, max_amplitude=mean_ramp_noise_value, noise_stdev=stdev_ramp_noise_value)
		
		# Generate a constant mean inhibitory noisy current for each MU - Noise is filtered at 150 Hz due to lowpass filtering properties of neuron membrane
		constant_inhibitory_bias_current_times, constant_inhibitory_noise_current_amp = generate_Noise_Signal(start_time=0.0, stop_time=simulation_duration+10, dt=1.0, mean_amplitude=mean_inhibition_noise_values[i], noise_stdev=stdev_inhibition_noise_values[i])
		
		# Generate the total (excitatory + inhibitory) noise bias current to the MU 
		total_bias_current_amp = ramped_noise_current_amp + constant_inhibitory_noise_current_amp
		
		# Neuron vector of MU dendritic bias currents
		#MU_Dendritic_Bias_Current_neuron.append([h.Vector(ramped_noise_current_amp), h.Vector(ramped_noise_current_amp), h.Vector(ramped_noise_current_amp), h.Vector(ramped_noise_current_amp)])
		MU_Dendritic_Bias_Current_neuron.append([h.Vector(total_bias_current_amp), h.Vector(total_bias_current_amp), h.Vector(total_bias_current_amp), h.Vector(total_bias_current_amp)])
		MU_Dendritic_Bias_times_neuron.append([h.Vector(ramped_bias_current_times), h.Vector(ramped_bias_current_times), h.Vector(ramped_bias_current_times), h.Vector(ramped_bias_current_times)])
		
		# Play the ramped bias current into eacb dendrite on the motoneurons
		MU_Dendritic_Bias_Current_neuron[i][0].play(GV.MU_dendritic_bias_current_iclamps[i][0]._ref_amp, MU_Dendritic_Bias_times_neuron[i][0], 1)	# Play current into first dendrite
		MU_Dendritic_Bias_Current_neuron[i][1].play(GV.MU_dendritic_bias_current_iclamps[i][1]._ref_amp, MU_Dendritic_Bias_times_neuron[i][1], 1)	# Play current into second dendrite
		MU_Dendritic_Bias_Current_neuron[i][2].play(GV.MU_dendritic_bias_current_iclamps[i][2]._ref_amp, MU_Dendritic_Bias_times_neuron[i][2], 1)	# Play current into third dendrite
		MU_Dendritic_Bias_Current_neuron[i][3].play(GV.MU_dendritic_bias_current_iclamps[i][3]._ref_amp, MU_Dendritic_Bias_times_neuron[i][3], 1)	# Play current into fourth dendrite
	
	# Set the basic noise conductance values - I think it may not actually be suitable
	Motoneuron_Pop.set(dendrite_Gfluct_ge0 = 0)
	Motoneuron_Pop.set(dendrite_Gfluct_stde = 0)
	Motoneuron_Pop.set(dendrite_Gfluct_gi0 = 0)
	Motoneuron_Pop.set(dendrite_Gfluct_stdi = 0)
	
	##############################################################################################################################################################
	##
	##																	DBS & Biomarker Setup
	##
	##############################################################################################################################################################
	# Beta State Observer + Controller Definitions 
	# ********************************************************
	# Window length for estimating beta biomarker
	beta_estimate_window_length = 2000.0																# ms
	beta_rec_sampling_interval = rec_sampling_interval													# Same as LFP signal sampling period - 0.5 ms
	beta_estimate_window_length_no_samples = int(beta_estimate_window_length/beta_rec_sampling_interval)
	
	# Window Tail length for estimating beta which is discarded post filtering - tail length is not used to estimate biomarker due to filter distortion on this tail
	beta_estimate_window_tail_length = 100.0				# ms	
	beta_estimate_window_tail_length_no_samples = int(beta_estimate_window_tail_length/beta_rec_sampling_interval)
	
	motor_symptom_controller_sampling_time = 20.0						# ms
	motor_symptom_controller_call_times = np.arange(steady_state_duration+beta_estimate_window_length+motor_symptom_controller_sampling_time, simulation_duration, motor_symptom_controller_sampling_time)
	
	# LFP Beta ARV state observer & controller definitions
	beta_ARV_Observer = State_Observer(Ts=motor_symptom_controller_sampling_time/1e3)
	beta_Biomarker_Amplitude_Controller = standard_PID_Controller(SetPoint=5.2070e-5, Kp=0.1138, Ti=0.2, Td=0, Ts=motor_symptom_controller_sampling_time/1e3, MinValue=0.0, MaxValue=3.0)

	# ********************************************************
	# Tremor & sEMG State Observers + Tremor Controller Definitions 
	# ********************************************************
	force_estimate_window_length = 2000               # ms
	force_rec_sampling_interval = 0.01					# ms
	force_estimate_window_length_no_samples = force_estimate_window_length/force_rec_sampling_interval
	
	# Tremor biomarker is estimated over the window tail length - after the taper on the end of the window is removed - taper is discarded (like the tail length in the beta estimate above) due to filtering artefact
	force_estimate_window_tail_length = 500.0                	# ms
	force_estimate_window_taper_length = 100.0                # ms
	force_estimate_window_tail_length_no_samples = force_estimate_window_tail_length/force_rec_sampling_interval
	force_estimate_window_taper_length_no_samples = force_estimate_window_taper_length/force_rec_sampling_interval
	
	# Force Tremor ARV state observer & controller definitions
	tremor_ARV_Observer = State_Observer(Ts=motor_symptom_controller_sampling_time/1e3)
	tremor_Biomarker_Amplitude_Controller = standard_PID_Controller(SetPoint=15.0, Kp=0.6271, Ti=1.0, Td=0, Ts=motor_symptom_controller_sampling_time/1e3, MinValue=0.0, MaxValue=3.0)

	# sEMG ARV state observer definition
	sEMG_ARV_Observer = State_Observer(Ts=motor_symptom_controller_sampling_time/1e3)
	
	# ********************************************************
	# Controller Bank & Scheduler Definitions
	# ********************************************************
	controller_bank = [tremor_Biomarker_Amplitude_Controller, beta_Biomarker_Amplitude_Controller]
	scheduler_bounds = [[0, 1.5e-5], [1.5e-5, 1e9]]											# Bounds are based on sEMG ARV - Tremor controller used if EMG ARV below 1.5e-5, beta if above
	control_scheduler = Controller_Scheduler(Controller_Bank=controller_bank, Scheduler_Bounds=scheduler_bounds, MinParameterValue=0.0, MaxParameterValue=3.0, RampDuration=0.25, Ts=0.02, Scheduler_Label='Scheduler', Controller_Labels=['Tremor_ARV', 'Beta_ARV'])
	
	# ********************************************************
	# Side-effects State Observer + Controller Definitions 
	# ********************************************************
	side_effect_estimate_window_length = 1000.0																# ms
	side_effect_rec_sampling_interval = motor_symptom_controller_sampling_time								# New DBS amplitude is calculated at each motor symptom controller call
	side_effect_estimate_window_length_no_samples = int(side_effect_estimate_window_length/side_effect_rec_sampling_interval)
	
	side_Effects_symptom_controller_sampling_time = 200.0						# ms
	side_Effects_Observer = State_Observer(Ts=side_Effects_symptom_controller_sampling_time/1e3)
	side_Effects_Avg_DBS_Amp_Observer = State_Observer(Ts=side_Effects_symptom_controller_sampling_time/1e3)
	# Side-Effect Controller with Ti = 1.2 and different setpoint tolerable amplitude values to avoid side-effects - 
	#side_Effects_Biomarker_Pulse_Duration_Controller = standard_PID_Controller_Plus_Bias(SetPoint=2.25, Kp=43.4128, Ti=1.2, Td=0, Bias=60.0, Ts=side_Effects_symptom_controller_sampling_time/1e3, MinValue=60.0, MaxValue=120.0)
	#side_Effects_Biomarker_Pulse_Duration_Controller = standard_PID_Controller_Plus_Bias(SetPoint=2.00, Kp=36.7629, Ti=1.2, Td=0, Bias=60.0, Ts=side_Effects_symptom_controller_sampling_time/1e3, MinValue=60.0, MaxValue=120.0)
	#side_Effects_Biomarker_Pulse_Duration_Controller = standard_PID_Controller_Plus_Bias(SetPoint=1.75, Kp=31.8796, Ti=1.2, Td=0, Bias=60.0, Ts=side_Effects_symptom_controller_sampling_time/1e3, MinValue=60.0, MaxValue=120.0)
	side_Effects_Biomarker_Pulse_Duration_Controller = standard_PID_Controller_Plus_Bias(SetPoint=1.5, Kp=28.1415, Ti=1.2, Td=0, Bias=60.0, Ts=side_Effects_symptom_controller_sampling_time/1e3, MinValue=60.0, MaxValue=120.0)
	#side_Effects_Biomarker_Pulse_Duration_Controller = standard_PID_Controller_Plus_Bias(SetPoint=1.25, Kp=25.1881, Ti=1.2, Td=0, Bias=60.0, Ts=side_Effects_symptom_controller_sampling_time/1e3, MinValue=60.0, MaxValue=120.0)

	# Track the implemented DBS amplitudes and calculate the average between side-effect controller calls 
	implemented_DBS_amplitudes = []
	
	# Define the side-effects controller call times 
	side_Effects_symptom_controller_call_times = np.arange(steady_state_duration+beta_estimate_window_length+side_effect_estimate_window_length, simulation_duration, side_Effects_symptom_controller_sampling_time)
	
	# ********************************************************
	# Initialize DBS signals for cortical collaterals & GPe
	# ********************************************************
	side_Effects_Biomarker_Output_DBS_Pulse_Width_value = DBS_pulse_width_value		# Initialize the side effects controller output value
	
	DBS_Signal, DBS_times, next_DBS_pulse_time = generate_DBS_Signal(start_time=steady_state_duration+10+simulator.state.dt, stop_time=simulation_duration, dt=simulator.state.dt,
													amplitude=-1.0, frequency=130.0, pulse_width=0.06, offset=0)	
	
	DBS_Signal = np.hstack((np.array([0, 0]), DBS_Signal))
	DBS_times = np.hstack((np.array([0, steady_state_duration+10]), DBS_times))
	
	# Set first portion of DBS signal (Up to first controller call after steady state) to zero amplitude
	DBS_Signal[0:] = 0
	next_DBS_pulse_time = motor_symptom_controller_call_times[0]
	
	DBS_Signal_neuron = h.Vector(DBS_Signal)
	DBS_times_neuron = h.Vector(DBS_times)
	
	# Play DBS signal to global variable is_xtra
	DBS_Signal_neuron.play(h._ref_is_xtra, DBS_times_neuron, 1)
	
	# Get DBS_Signal_neuron as a numpy array for easy updating
	updated_DBS_signal = DBS_Signal_neuron.as_numpy()
	
	# 2-D Interpolation Function where the DBS amplitude and pulse width both determine the number of collaterals entrained during stimulation
	interp_DBS_amplitudes = np.loadtxt('DBS_Amplitude_Interpolation_values.txt', delimiter=',')			
	interp_DBS_pulse_widths = np.loadtxt('DBS_Pulse_Width_Interpolation_values.txt', delimiter=',')	
	interp_collaterals_entrained = np.loadtxt('Collaterals_Entrained_values.txt', delimiter=',')	
	# Make the 2-D interpolation function object to use during simulation
	activated_GPe_interpolation = interpolate.RectBivariateSpline(x=interp_DBS_amplitudes, y=interp_DBS_pulse_widths, z=interp_collaterals_entrained, bbox=[None, None, None, None], kx=3, ky=3, s=0)

	# Load the GPe Stimulation Order
	GPe_stimulation_order = np.loadtxt('GPe_Stimulation_Order.txt', delimiter=',')
	GPe_stimulation_order = [int(index) for index in GPe_stimulation_order]

	# Make new GPe DBS vector for each GPe neuron - each GPe neuron needs a pointer to it's own DBS signal
	GPe_DBS_Signal_neuron = []
	GPe_DBS_times_neuron = []
	updated_GPe_DBS_signal = []
	for i in range(0, Pop_size):
	
		GPe_DBS_Signal, GPe_DBS_times, GPe_next_DBS_pulse_time = generate_DBS_Signal(start_time=steady_state_duration+10+simulator.state.dt, stop_time=simulation_duration, dt=simulator.state.dt,
														amplitude=100.0, frequency=130.0, pulse_width=0.06, offset=0)	
		
		GPe_DBS_Signal = np.hstack((np.array([0, 0]), GPe_DBS_Signal))
		GPe_DBS_times = np.hstack((np.array([0, steady_state_duration+10]), GPe_DBS_times))
		
		# Set the GPe DBS signals to zero amplitude
		GPe_DBS_Signal[0:] = 0
		GPe_next_DBS_pulse_time = motor_symptom_controller_call_times[0]
		
		# Neuron vector of GPe DBS signals
		GPe_DBS_Signal_neuron.append(h.Vector(GPe_DBS_Signal))
		GPe_DBS_times_neuron.append(h.Vector(GPe_DBS_times))
		
		# Play the stimulation into eacb GPe neuron
		GPe_DBS_Signal_neuron[i].play(GV.GPe_stimulation_iclamps[i]._ref_amp, GPe_DBS_times_neuron[i], 1)
		
		# Hold a reference to the signal as a numpy array, and append to list of GPe stimulation signals
		updated_GPe_DBS_signal.append(GPe_DBS_Signal_neuron[i].as_numpy())
	
	# Define state observers for tracking the stimulation amplitude and pulse duration which are applied at each time in the model
	stimulation_Amplitude_Observer = State_Observer(Ts=motor_symptom_controller_sampling_time/1e3)
	stimulation_Pulse_Duration_Observer = State_Observer(Ts=motor_symptom_controller_sampling_time/1e3)
	
	# Variables for writing simulation data
	last_write_time = steady_state_duration
	
	# ********************************************************
	# Initialize STN LFP, MU Force & MU sEMG signals 
	# ********************************************************
	# STN LFP Initialization - placeholder for LFP output signals 
	STN_LFP = []
	STN_LFP_AMPA = []
	STN_LFP_GABAa = []
	
	# Force Initialization - placeholder for force output signals
	force_time_signal = np.arange(0, simulation_duration+rec_sampling_interval, 0.01)	# time signal with same sampling frequency as model timestep
	force_time_signal = np.around(force_time_signal, 2)									# make sure resolution is only two decimal places
	force_signal = np.zeros(len(force_time_signal)).flatten()							# Force signal initialized as zeros vector
	
	# Generate the MU Pool force twitch parameters and set the intial spike times (i.e. intial interpulse interval is taken from steady state time)
	twitch_duration = 500							# ms
	P_values, T_values = generate_MU_Pool_Force_Twitch_Parameters(Pop_size=Motoneuron_Pop.size, R_P=100.0, T_max=90.0)
	last_MU_spike_times = steady_state_duration*np.ones(Motoneuron_Pop.size)			# Set initial spike times for calculating the force
	
	# sEMG Initialization - placeholder for sEMG output signal
	MU_Action_Potential_Shapes = np.load('MU_AP_Shapes.npy')							# Load the MU action potential shapes
	sEMG_shape_duration = 0.01*len(MU_Action_Potential_Shapes[0]) 						# 0.01 is the timestep for the models - need the MU shape duration for later
	
	sEMG_time_signal = np.arange(0, simulation_duration+rec_sampling_interval, 0.01)	# time signal with same sampling frequency as model timestep
	sEMG_time_signal = np.around(sEMG_time_signal, 2)									# make sure resolution is only two decimal places
	sEMG_signal = np.zeros(len(sEMG_time_signal)).flatten()								# Force signal initialized as zeros vector
	
	##############################################################################################################################################################
	##
	##																Simulate the model from SS
	##
	##############################################################################################################################################################
	
	"""
	# Run the model to the steady state
	run_until(steady_state_duration+simulator.state.dt)
	
	# Clear the data to this point
	Cortical_Pop.get_data('soma(0.5).v', clear=True)
	Interneuron_Pop.get_data('soma(0.5).v', clear=True)
	STN_Pop.get_data('soma(0.5).v', clear=True)
	GPe_Pop.get_data('soma(0.5).v', clear=True)
	GPi_Pop.get_data('soma(0.5).v', clear=True)
	Thalamic_Pop.get_data('soma(0.5).v', clear=True)
	Motoneuron_Pop.get_data('soma(0.5).v', clear=True)
	"""
	
	# Load the steady state
	run_until(steady_state_duration+simulator.state.dt, run_from_steady_state=True)
	
	# Reload striatal spike times
	for i in range(0,Pop_size):
		Striatal_Pop[i].spike_times=striatal_spike_times[i][0]
	
	# Reload the independent beta spike times
	for i in range(0,6*Pop_size):
		Independent_Motoneuron_Input_Pop[i].spike_times=independent_beta_spike_times[i][0]
	
	# Reload the Thalamic -> Cortical synaptic connection weights
	prj_ThalamicCortical.set(weight=g_Thalamic_Cortical_weights)
	
	# For loop to integrate the model state up to each motor symptom controller call time - motor symptom controller is more frequent than side-effects controller, so it is used to integrate the model
	for motor_symptom_controller_call_index, motor_symptom_controller_call_time in enumerate(motor_symptom_controller_call_times):

		# Integrate model to motor_symptom_controller_call_time
		run_until(motor_symptom_controller_call_time-simulator.state.dt)
		
		# Check if the parallel side-effects controller is also called at this time:
		if motor_symptom_controller_call_time in side_Effects_symptom_controller_call_times:
			print(("Side-Effects Controller Called at t: %f" % simulator.state.t))
			
			# Calculate the average of the DBS amplitude over the last 1 s window of calls to the motor symptoms controllers
			average_DBS_amplitude = np.mean(implemented_DBS_amplitudes[-side_effect_estimate_window_length_no_samples:])
			side_Effects_Avg_DBS_Amp_Observer.update(state_value=average_DBS_amplitude, current_time=simulator.state.t)
			
			# Update the mean SS side-effects observer and pulse duration controller 
			side_Effects_Observer.update(state_value=average_DBS_amplitude, current_time=simulator.state.t)
			side_Effects_Biomarker_Output_DBS_Pulse_Width_value = side_Effects_Biomarker_Pulse_Duration_Controller.update(state_value=average_DBS_amplitude, current_time=simulator.state.t)
			
		print(("Motor Symptoms Controllers Called at t: %f" % simulator.state.t))
		
		# Calculate the LFP and biomarkers, etc. 
		STN_AMPA_i = np.array(STN_Pop.get_data('AMPA.i').segments[0].analogsignals[0])
		STN_GABAa_i = np.array(STN_Pop.get_data('GABAa.i').segments[0].analogsignals[0])
		STN_Syn_i = STN_AMPA_i + STN_GABAa_i
		
		# STN LFP Calculation - Syn_i is in units of nA -> LFP units are mV
		STN_LFP_1 = (1/(4*math.pi*sigma))*np.sum((1/(STN_recording_electrode_1_distances*1e-6))*STN_Syn_i.transpose(),axis=0)*1e-6 
		STN_LFP_2 = (1/(4*math.pi*sigma))*np.sum((1/(STN_recording_electrode_2_distances*1e-6))*STN_Syn_i.transpose(),axis=0)*1e-6 
		STN_LFP = np.hstack((STN_LFP, STN_LFP_1 - STN_LFP_2))	
			
		# STN LFP AMPA and GABAa Contributions
		STN_LFP_AMPA_1 = (1/(4*math.pi*sigma))*np.sum((1/(STN_recording_electrode_1_distances*1e-6))*STN_AMPA_i.transpose(),axis=0)*1e-6 
		STN_LFP_AMPA_2 = (1/(4*math.pi*sigma))*np.sum((1/(STN_recording_electrode_2_distances*1e-6))*STN_AMPA_i.transpose(),axis=0)*1e-6 
		STN_LFP_AMPA = np.hstack((STN_LFP_AMPA, STN_LFP_AMPA_1 - STN_LFP_AMPA_2))	
		STN_LFP_GABAa_1 = (1/(4*math.pi*sigma))*np.sum((1/(STN_recording_electrode_1_distances*1e-6))*STN_GABAa_i.transpose(),axis=0)*1e-6 
		STN_LFP_GABAa_2 = (1/(4*math.pi*sigma))*np.sum((1/(STN_recording_electrode_2_distances*1e-6))*STN_GABAa_i.transpose(),axis=0)*1e-6 
		STN_LFP_GABAa = np.hstack((STN_LFP_GABAa, STN_LFP_GABAa_1 - STN_LFP_GABAa_2))	
		
		# Beta Biomarker Calculation:
		lfp_beta_ARV = calculate_beta_ARV(lfp_signal=STN_LFP[-beta_estimate_window_length_no_samples:], tail_length=beta_estimate_window_tail_length_no_samples, beta_b=beta_b, beta_a=beta_a)
		
		# Update the Beta Biomarker Observer & calculate potential DBS amplitude value using the beta biomarker
		beta_ARV_Observer.update(state_value=lfp_beta_ARV, current_time=simulator.state.t)
		
		# Calculate the updated force signal
		MU_Pool_Spiketrains = Motoneuron_Pop.get_data('spikes').segments[0].spiketrains		# Get the spike times from the pool
		force_time_signal, force_signal, last_MU_spike_times = calculate_Total_Pool_Force(t=force_time_signal, total_pool_force_signal=force_signal, pool_spike_trains=MU_Pool_Spiketrains, P_values=P_values, T_values=T_values, twitch_duration=twitch_duration, dt=0.01, initial_spike_times=last_MU_spike_times)
		
		# Update the force biomarkers
		current_time_index = np.where(force_time_signal==motor_symptom_controller_call_time)[0][0]		# Find the current Force signal window
		force_signal_segment = force_signal[int(current_time_index-force_estimate_window_length_no_samples):current_time_index]
		
		# Tremor (Force) Biomarker Calculation:
		tremor_ARV = calculate_tremor_ARV(force_signal=force_signal_segment, estimate_window_length=int(force_estimate_window_tail_length_no_samples), tail_length=int(force_estimate_window_taper_length_no_samples), tremor_b=tremor_b, tremor_a=tremor_a)
		
		# Update the Force Tremor ARV Biomarker Observer & calculate potential DBS amplitude value using the biomarker
		tremor_ARV_Observer.update(state_value=tremor_ARV, current_time=simulator.state.t)
		
		# Calculate the updated sEMG Signal
		sEMG_time_signal, sEMG_signal = calculate_MU_Pool_sEMG(t=sEMG_time_signal, total_pool_sEMG_signal=sEMG_signal, pool_spike_trains=MU_Pool_Spiketrains, MU_Action_Potential_shapes=MU_Action_Potential_Shapes, sEMG_shape_duration=sEMG_shape_duration, dt=0.01)
		
		# Update the force biomarkers
		current_time_index = np.where(sEMG_time_signal==motor_symptom_controller_call_time)[0][0]		# Find the current Force signal window
		sEMG_signal_segment = sEMG_signal[int(current_time_index-force_estimate_window_length_no_samples):current_time_index]
		
		# Muscle Activation Biomarker Calculation:
		sEMG_ARV = calculate_sEMG_ARV(sEMG_signal=sEMG_signal_segment, estimate_window_length=int(force_estimate_window_tail_length_no_samples))	# Estimate the sEMG power on 500 ms window like the force biomarkers
		
		# Update the sEMG ARV observer
		sEMG_ARV_Observer.update(state_value=sEMG_ARV, current_time=simulator.state.t)
		
		# Classify the muscle activation state here and update the control scheduler-
		scheduler_Output_DBS_amplitude_value = control_scheduler.update(scheduler_state_value=sEMG_ARV, controller_state_values=[tremor_ARV, lfp_beta_ARV], current_time=simulator.state.t)
		
		# Set the actual DBS parameters (amplitude & pulse duration) to implement
		DBS_amplitude_value = scheduler_Output_DBS_amplitude_value 			# Use the control scheduler output amplitude value
		#DBS_pulse_width_value = np.round(side_Effects_Biomarker_Output_DBS_Pulse_Width_value, 2)
		DBS_pulse_width_value = np.round(side_Effects_Biomarker_Output_DBS_Pulse_Width_value/1.0e3, 2)		# Convert from us to ms
		
		# Track the implemented DBS amplitudes for the side-effects controller
		implemented_DBS_amplitudes.append(DBS_amplitude_value)
		
		# Update the implemented DBS parameter state observers
		stimulation_Amplitude_Observer.update(state_value=DBS_amplitude_value, current_time=simulator.state.t)
		stimulation_Pulse_Duration_Observer.update(state_value=DBS_pulse_width_value, current_time=simulator.state.t)
		
		# Update the DBS Signal
		if motor_symptom_controller_call_index+1 < len(motor_symptom_controller_call_times):
			
			if next_DBS_pulse_time < motor_symptom_controller_call_times[motor_symptom_controller_call_index+1]:
				
				GPe_next_DBS_pulse_time = next_DBS_pulse_time
				
				# DBS Cortical Collateral Stimulation
				new_DBS_Signal_Segment, new_DBS_times_Segment, next_DBS_pulse_time = generate_DBS_Signal(start_time=next_DBS_pulse_time, stop_time=motor_symptom_controller_call_times[motor_symptom_controller_call_index+1], dt=simulator.state.dt,
													amplitude=-DBS_amplitude_value, frequency=130.0, pulse_width=DBS_pulse_width_value, offset=0)	
					
				# Update DBS segment - replace original DBS array values with updated ones
				window_start_index = np.where(DBS_times==new_DBS_times_Segment[0])[0][0]
				new_window_sample_length = len(new_DBS_Signal_Segment)
				updated_DBS_signal[window_start_index:window_start_index+new_window_sample_length] = new_DBS_Signal_Segment
				
				# DBS GPe neuron stimulation
				# 2-D Interpolation
				num_GPe_Neurons_entrained = int(activated_GPe_interpolation.ev(DBS_amplitude_value, DBS_pulse_width_value))
				
				# Limit the max and min no. of GPe neurons entrained to the population size
				if num_GPe_Neurons_entrained > Pop_size:
					num_GPe_Neurons_entrained = Pop_size
				elif num_GPe_Neurons_entrained < 0:
					num_GPe_Neurons_entrained = 0
				
				# Make copy of current DBS segment and rescale for GPe neuron stimulation
				GPe_DBS_Signal_Segment = new_DBS_Signal_Segment.copy()
				GPe_DBS_Signal_Segment *= -1
				GPe_DBS_Signal_Segment[GPe_DBS_Signal_Segment>0] = 100
				
				# Stimulate the entrained GPe neurons
				for i in np.arange(0, num_GPe_Neurons_entrained):
					updated_GPe_DBS_signal[GPe_stimulation_order[i]][window_start_index:window_start_index+new_window_sample_length] = GPe_DBS_Signal_Segment

			else:
				pass
				
		# Write population data to file
		simulation_label = "Model_Results"
		write_index = "{:.0f}_".format(motor_symptom_controller_call_index)
		suffix = "_{:.0f}ms-{:.0f}ms".format(last_write_time, simulator.state.t)
		
		# Write STN membrane voltage to file
		STN_Pop.write_data(simulation_label+"/STN_Pop/"+write_index+"STN_Soma_v"+suffix+".mat", 'soma(0.5).v', clear=True)
		
		# Write motoneuron pool soma membrane voltage and spike times to file
		Motoneuron_Pop.write_data(simulation_label+"/Motoneuron_Pop/"+write_index+"Motoneuron_Spike_times"+suffix+".mat", 'spikes', clear=False)
		Motoneuron_Pop.write_data(simulation_label+"/Motoneuron_Pop/"+write_index+"Motoneuron_Soma_v"+suffix+".mat", 'soma(0.5).v', clear=True)
	
		last_write_time = simulator.state.t
	
	# Stop the scheduler when simulation is done
	control_scheduler.stop()
	
	# Write the scheduler data to file
	scheduler_state_values = np.asarray(control_scheduler.get_scheduler_state_history())	
	scheduler_output_values = np.asarray(control_scheduler.get_scheduler_output_history())
	scheduler_classification_values = np.asarray(control_scheduler.get_scheduler_classification_history())
	scheduler_sample_times = np.asarray(control_scheduler.get_scheduler_sample_times())
	np.savetxt(simulation_label+"/scheduler_state_values.csv", scheduler_state_values, delimiter=',')
	np.savetxt(simulation_label+"/scheduler_output_values.csv", scheduler_output_values, delimiter=',')
	np.savetxt(simulation_label+"/scheduler_classification_values.csv", scheduler_classification_values, delimiter=',')
	np.savetxt(simulation_label+"/scheduler_sample_times.csv", scheduler_sample_times, delimiter=',')
	
	# Write the scheduler controller bank data to file
	controller_bank_state_values = np.asarray(control_scheduler.get_controller_state_history())
	controller_bank_error_values = np.asarray(control_scheduler.get_controller_error_history())
	controller_bank_output_values = np.asarray(control_scheduler.get_controller_output_history())
	controller_bank_sample_times = np.asarray(control_scheduler.get_controller_sample_times())
	
	# Get the controller labels
	controller_bank_controller_labels = control_scheduler.get_Controller_Labels()

	# Write the state of each controller in the bank to file 
	for controller_id in np.arange(0, len(controller_bank_state_values)):
		controller_label = controller_bank_controller_labels[controller_id]
		for control_period_id in np.arange(0,len(controller_bank_state_values[controller_id])):
			output_filename = simulation_label+"/Controller_Bank_" + controller_label + "_Controller_Period_{:.0f}_state_values.csv".format(control_period_id+1) 
			np.savetxt(output_filename, controller_bank_state_values[controller_id][control_period_id], delimiter=',')

	# Write the error of each controller in the bank to file 
	for controller_id in np.arange(0, len(controller_bank_error_values)):
		controller_label = controller_bank_controller_labels[controller_id]
		for control_period_id in np.arange(0,len(controller_bank_error_values[controller_id])):
			output_filename = simulation_label+"/Controller_Bank_" + controller_label + "_Controller_Period_{:.0f}_error_values.csv".format(control_period_id+1) 
			np.savetxt(output_filename, controller_bank_error_values[controller_id][control_period_id], delimiter=',')

	# Write the output of each controller in the bank to file 
	for controller_id in np.arange(0, len(controller_bank_output_values)):
		controller_label = controller_bank_controller_labels[controller_id]
		for control_period_id in np.arange(0,len(controller_bank_output_values[controller_id])):
			output_filename = simulation_label+"/Controller_Bank_" + controller_label + "_Controller_Period_{:.0f}_output_values.csv".format(control_period_id+1) 
			np.savetxt(output_filename, controller_bank_output_values[controller_id][control_period_id], delimiter=',')

	# Write the sample times of each controller in the bank to file 
	for controller_id in np.arange(0, len(controller_bank_sample_times)):
		controller_label = controller_bank_controller_labels[controller_id]
		for control_period_id in np.arange(0,len(controller_bank_sample_times[controller_id])):
			output_filename = simulation_label+"/Controller_Bank_" + controller_label + "_Controller_Period_{:.0f}_sample_times.csv".format(control_period_id+1) 
			np.savetxt(output_filename, controller_bank_sample_times[controller_id][control_period_id], delimiter=',')
	
	# Write State Observers and Controllers values to csv files
	# Beta Biomarker Observer - 
	beta_ARV_Observer_values = np.asarray(beta_ARV_Observer.get_state_history())
	beta_ARV_Observer_times = np.asarray(beta_ARV_Observer.get_sample_times())
	np.savetxt(simulation_label+"/beta_ARV_Observer_values.csv", beta_ARV_Observer_values, delimiter=',')
	np.savetxt(simulation_label+"/Motor_Symptom_Sample_Times.csv", beta_ARV_Observer_times, delimiter=',')
	
	# Force Tremor Observers - 
	tremor_ARV_Observer_values = np.asarray(tremor_ARV_Observer.get_state_history())
	tremor_ARV_Observer_times = np.asarray(tremor_ARV_Observer.get_sample_times())
	np.savetxt(simulation_label+"/tremor_ARV_Observer_values.csv", tremor_ARV_Observer_values, delimiter=',')
	
	# sEMG ARV Observer -
	sEMG_ARV_Observer_values = np.asarray(sEMG_ARV_Observer.get_state_history())
	sEMG_ARV_Observer_times = np.asarray(sEMG_ARV_Observer.get_sample_times())
	np.savetxt(simulation_label+"/sEMG_ARV_Observer_values.csv", sEMG_ARV_Observer_values, delimiter=',')
	np.savetxt(simulation_label+"/sEMG_ARV_Observer_times.csv", sEMG_ARV_Observer_times, delimiter=',')
	
	# Side-Effects Biomarker Observer - 
	side_Effects_Observer_values = np.asarray(side_Effects_Observer.get_state_history())
	side_Effects_Avg_DBS_Amp_Observer_values = np.asarray(side_Effects_Avg_DBS_Amp_Observer.get_state_history())
	side_Effects_Observer_times = np.asarray(side_Effects_Observer.get_sample_times())
	np.savetxt(simulation_label+"/side_Effects_Observer_values.csv", side_Effects_Observer_values, delimiter=',')
	np.savetxt(simulation_label+"/side_Effects_Avg_DBS_Amp_Observer_values.csv", side_Effects_Avg_DBS_Amp_Observer_values, delimiter=',')
	np.savetxt(simulation_label+"/side_Effects_Observer_times.csv", side_Effects_Observer_times, delimiter=',')
	np.savetxt(simulation_label+"/Side_Effects_Symptom_Sample_Times.csv", side_Effects_Observer_times, delimiter=',')
	
	# Side-Effects Biomarker Controller - 
	side_Effects_Biomarker_Controller_Output_DBS_pulse_duration_values = np.asarray(side_Effects_Biomarker_Pulse_Duration_Controller.get_output_history())
	side_Effects_Biomarker_Controller_State_values = np.asarray(side_Effects_Biomarker_Pulse_Duration_Controller.get_state_history())
	side_Effects_Biomarker_Controller_Error_values = np.asarray(side_Effects_Biomarker_Pulse_Duration_Controller.get_error_history())
	side_Effects_Biomarker_Controller_times = np.asarray(side_Effects_Biomarker_Pulse_Duration_Controller.get_sample_times())
	np.savetxt(simulation_label+"/side_Effects_Controller_Output_DBS_Pulse_Duration_values.csv", side_Effects_Biomarker_Controller_Output_DBS_pulse_duration_values, delimiter=',')
	np.savetxt(simulation_label+"/side_Effects_Controller_State_values.csv", side_Effects_Biomarker_Controller_State_values, delimiter=',')
	np.savetxt(simulation_label+"/side_Effects_Controller_Error_values.csv", side_Effects_Biomarker_Controller_Error_values, delimiter=',')
	np.savetxt(simulation_label+"/side_Effects_Controller_times.csv", side_Effects_Biomarker_Controller_times, delimiter=',')
	
	# Implemented DBS Parameter Observers - 
	stimulation_Amplitude_Observer_values = np.asarray(stimulation_Amplitude_Observer.get_state_history())
	stimulation_Amplitude_Observer_times = np.asarray(stimulation_Amplitude_Observer.get_sample_times())
	stimulation_Pulse_Duration_Observer_values = np.asarray(stimulation_Pulse_Duration_Observer.get_state_history())
	stimulation_Pulse_Duration_Observer_times = np.asarray(stimulation_Pulse_Duration_Observer.get_sample_times())
	np.savetxt(simulation_label+"/stimulation_Amplitude_Observer_values.csv", stimulation_Amplitude_Observer_values, delimiter=',')
	np.savetxt(simulation_label+"/stimulation_Amplitude_Observer_times.csv", stimulation_Amplitude_Observer_times, delimiter=',')
	np.savetxt(simulation_label+"/stimulation_Pulse_Duration_Observer_values.csv", stimulation_Pulse_Duration_Observer_values, delimiter=',')
	np.savetxt(simulation_label+"/stimulation_Pulse_Duration_Observer_times.csv", stimulation_Pulse_Duration_Observer_times, delimiter=',')
	
	# Write population membrane voltage data to file -  uncomment if interested in these values
	#Cortical_Pop.write_data(simulation_label+"/Cortical_Pop/Cortical_Collateral_v.mat", 'collateral(0.5).v', clear=False)
	#Cortical_Pop.write_data(simulation_label+"/Cortical_Pop/Cortical_Soma_v.mat", 'soma(0.5).v', clear=False)
	#Cortical_Pop.write_data(simulation_label+"/Cortical_Pop/Cortical_End_Node_v.mat", 'end_node(0.5).v', clear=False)
	#Cortical_Pop.write_data(simulation_label+"/Cortical_Pop/Cortical_Soma_AMPA_i.mat", 'AMPA.i', clear=False)
	#Cortical_Pop.write_data(simulation_label+"/Cortical_Pop/Cortical_Soma_GABAa_i.mat", 'GABAa.i', clear=True)
	#Interneuron_Pop.write_data(simulation_label+"/Interneuron_Pop/Interneuron_Soma_v.mat", 'soma(0.5).v', clear=True)
	#STN_Pop.write_data(simulation_label+"/STN_Pop/STN_Soma_v.mat", 'soma(0.5).v', clear=True)
	#GPe_Pop.write_data(simulation_label+"/GPe_Pop/GPe_Soma_v.mat", 'soma(0.5).v', clear=True)
	#GPi_Pop.write_data(simulation_label+"/GPi_Pop/GPi_Soma_v.mat", 'soma(0.5).v', clear=True)
	#Thalamic_Pop.write_data(simulation_label+"/Thalamic_Pop/Thalamic_Soma_v.mat", 'soma(0.5).v', clear=False)
	#Thalamic_Pop.write_data(simulation_label+"/Thalamic_Pop/Thalamic_AMPA_i.mat", 'GABAa.i', clear=False)
	#Thalamic_Pop.write_data(simulation_label+"/Thalamic_Pop/Thalamic_GABAa_i.mat", 'GABAa.i', clear=True)
	
	
	# Write Independent inputs to MU pool to file
	#Independent_Motoneuron_Input_Pop.write_data(simulation_label+"/Independent_Pop/Independent_Spike_times.mat", 'spikes', clear=False)
	
	# Write the STN LFP to .mat file
	STN_LFP_Block = neo.Block(name='STN_LFP')
	STN_LFP_seg = neo.Segment(name='segment_0')
	STN_LFP_Block.segments.append(STN_LFP_seg)
	STN_LFP_signal = neo.AnalogSignal(STN_LFP, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
	STN_LFP_seg.analogsignals.append(STN_LFP_signal)
	
	w = neo.io.NeoMatlabIO(filename=simulation_label+"/STN_LFP.mat") 
	w.write_block(STN_LFP_Block)
	
	# Write LFP AMPA and GABAa conmponents to file
	STN_LFP_AMPA_Block = neo.Block(name='STN_LFP_AMPA')
	STN_LFP_AMPA_seg = neo.Segment(name='segment_0')
	STN_LFP_AMPA_Block.segments.append(STN_LFP_AMPA_seg)
	STN_LFP_AMPA_signal = neo.AnalogSignal(STN_LFP_AMPA, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
	STN_LFP_AMPA_seg.analogsignals.append(STN_LFP_AMPA_signal)
	w = neo.io.NeoMatlabIO(filename=simulation_label+"/STN_LFP_AMPA.mat") 
	w.write_block(STN_LFP_AMPA_Block)
	
	STN_LFP_GABAa_Block = neo.Block(name='STN_LFP_GABAa')
	STN_LFP_GABAa_seg = neo.Segment(name='segment_0')
	STN_LFP_GABAa_Block.segments.append(STN_LFP_GABAa_seg)
	STN_LFP_GABAa_signal = neo.AnalogSignal(STN_LFP_GABAa, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
	STN_LFP_GABAa_seg.analogsignals.append(STN_LFP_GABAa_signal)
	w = neo.io.NeoMatlabIO(filename=simulation_label+"/STN_LFP_GABAa.mat") 
	w.write_block(STN_LFP_GABAa_Block)
	
	# Extract only the simulation period for the force and sEMG signals
	simulation_SS_index = np.where(force_time_signal>=steady_state_duration)[0][0]
	force_signal = force_signal[simulation_SS_index::]
	force_time_signal = force_time_signal[simulation_SS_index::]
	sEMG_signal = sEMG_signal[simulation_SS_index::]
	sEMG_time_signal = sEMG_time_signal[simulation_SS_index::]
	
	# Write force output signal to file
	np.savetxt(simulation_label+"/Force_amplitude_values.csv", force_signal, delimiter=',')
	np.savetxt(simulation_label+"/Force_times.csv", force_time_signal, delimiter=',')
	
	# Write sEMG output signal to file
	np.savetxt(simulation_label+"/sEMG_values.csv", sEMG_signal, delimiter=',')
	np.savetxt(simulation_label+"/sEMG_times.csv", sEMG_time_signal, delimiter=',')
	
	# Write the DBS Signal to .mat file
	# DBS Amplitude
	DBS_Block = neo.Block(name='DBS_Signal')
	DBS_Signal_seg = neo.Segment(name='segment_0')
	DBS_Block.segments.append(DBS_Signal_seg)
	DBS_signal = neo.AnalogSignal(DBS_Signal_neuron, units='mA', t_start=0*pq.ms, sampling_rate=pq.Quantity(1.0/simulator.state.dt, '1/ms'))
	DBS_Signal_seg.analogsignals.append(DBS_signal)
	DBS_times = neo.AnalogSignal(DBS_times_neuron, units='ms', t_start=DBS_times_neuron*pq.ms, sampling_rate=pq.Quantity(1.0/simulator.state.dt, '1/ms'))
	DBS_Signal_seg.analogsignals.append(DBS_times)
	
	w = neo.io.NeoMatlabIO(filename=simulation_label+"/DBS_Signal.mat") 
	w.write_block(DBS_Block)
	
	print("Full CBG Network + Motoneuron Pool Tremor Biomarker Simulation Done!")
	
	end()
