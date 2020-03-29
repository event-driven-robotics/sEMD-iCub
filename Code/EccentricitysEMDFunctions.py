#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:17:25 2019

@author: giulia.dangelo@iit.it
"""

from EccentricityFuncs import *
import pickle as pickle
import matplotlib.pyplot as plt
import re
import numpy as np
import random


class Event:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.timestamp = 0
        self.polarity = 0


def FakeStimuliBarMoving(camera_resolution, bar_speed, space_metric, bar_dimensions, direction, contrast):
    time = float(space_metric) / float(bar_speed)  # dt between one coloum an another one
    timestamp = 0
    eventsON = []
    eventsOFF = []

    if contrast == 'BlackOverWhite':
        polarity_front = 0
        polarity_back = 1
    elif contrast == 'WhiteOverBlack':
        polarity_front = 1
        polarity_back = 0

    if direction == 'LR':
        surface = np.zeros(camera_resolution)
        # place the bar in the center
        ybar_start = ((camera_resolution[1] / 2) - (bar_dimensions[1] / 2))
        ybar_end = ybar_start + bar_dimensions[1]

        y_length = ybar_end - ybar_start
        y_random = random.sample(range(ybar_start, ybar_end), y_length)

        for x in range(0, camera_resolution[0]):
            timestamp = timestamp + time
            for y in y_random:

                if (x - bar_dimensions[0] > 0):

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x - bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)

                    surface[y][x] = 255  # HERE
                else:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

    elif direction == 'RL':

        # place the bar in the center
        ybar_start = ((camera_resolution[1] / 2) - (bar_dimensions[1] / 2))
        ybar_end = ybar_start + bar_dimensions[1]

        y_length = ybar_end - ybar_start
        y_random = random.sample(range(ybar_start, ybar_end), y_length)

        for x in range(camera_resolution[0], 0, -1):
            timestamp = timestamp + time
            for y in y_random:
                if (x - bar_dimensions[0] > 0):

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x - bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                else:

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)


    elif direction == 'TB':

        # place the bar in the center
        xbar_start = ((camera_resolution[0] / 2) - (bar_dimensions[0] / 2))
        xbar_end = xbar_start + bar_dimensions[0]

        x_length = xbar_end - xbar_start
        x_random = random.sample(range(xbar_start, xbar_end), x_length)

        for y in range(0, camera_resolution[1]):
            timestamp = timestamp + time
            for x in x_random:
                if (y - bar_dimensions[1] > 0):

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x - bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                else:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

    elif direction == 'BT':

        # place the bar in the center
        xbar_start = ((camera_resolution[0] / 2) - (bar_dimensions[0] / 2))
        xbar_end = xbar_start + bar_dimensions[0]

        x_length = xbar_end - xbar_start
        x_random = random.sample(range(xbar_start, xbar_end), x_length)

        for y in range(camera_resolution[1], 0, -1):
            timestamp = timestamp + time
            for x in x_random:
                if (y - bar_dimensions[1] > 0):

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x - bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                else:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

    elif direction == 'TransTlBr':

        # place the bar in the center
        ybar_start = ((camera_resolution[1] / 2) - (bar_dimensions[1] / 2))
        ybar_end = ybar_start + bar_dimensions[1]

        y_length = ybar_end - ybar_start
        y_random = random.sample(range(ybar_start, ybar_end), y_length)
        time = float(space_metric) * np.sqrt(2) / float(bar_speed)  # dt between one coloum an another one
        time_off = float(space_metric) * bar_dimensions[0] / float(bar_speed)

        for run in range(camera_resolution[0] + camera_resolution[1]):
            if run < camera_resolution[0]:
                x = run
                y = 0
                while x >= 0 and y < camera_resolution[1]:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x - 1
                    y = y + 1
            else:
                x = camera_resolution[0] - 1
                y = run - camera_resolution[0]
                while x >= 0 and y < camera_resolution[1]:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x - 1
                    y = y + 1

            timestamp = timestamp + time
    elif direction == 'TransTrBl':

        # place the bar in the center
        ybar_start = ((camera_resolution[1] / 2) - (bar_dimensions[1] / 2))
        ybar_end = ybar_start + bar_dimensions[1]

        y_length = ybar_end - ybar_start
        y_random = random.sample(range(ybar_start, ybar_end), y_length)
        time = float(space_metric) * np.sqrt(2) / float(bar_speed)  # dt between one coloum an another one
        time_off = float(space_metric) * bar_dimensions[0] / float(bar_speed)

        for run in range(camera_resolution[0] + camera_resolution[1]):
            if run < camera_resolution[0]:
                x = camera_resolution[0] - 1 - run
                y = 0
                while x < camera_resolution[0] and y < camera_resolution[1]:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x + 1
                    y = y + 1
            else:
                x = 0
                y = run - camera_resolution[0]
                while x < camera_resolution[0] and y < camera_resolution[1]:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x + 1
                    y = y + 1

            timestamp = timestamp + time


    elif direction == 'TransBlTr':

        # place the bar in the center
        ybar_start = ((camera_resolution[1] / 2) - (bar_dimensions[1] / 2))
        ybar_end = ybar_start + bar_dimensions[1]

        y_length = ybar_end - ybar_start
        y_random = random.sample(range(ybar_start, ybar_end), y_length)
        time = float(space_metric) * np.sqrt(2) / float(bar_speed)  # dt between one coloum an another one
        time_off = float(space_metric) * bar_dimensions[0] / float(bar_speed)

        for run in range(camera_resolution[0] + camera_resolution[1]):
            if run < camera_resolution[0]:
                x = run
                y = camera_resolution[1] - 1
                while x >= 0 and y >= 0:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x - 1
                    y = y - 1
            else:
                x = camera_resolution[0] - 1
                y = camera_resolution[0] + camera_resolution[1] - run - 1
                while x >= 0 and y >= 0:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x - 1
                    y = y - 1

            timestamp = timestamp + time

    elif direction == 'TransBrTl':

        # place the bar in the center
        ybar_start = ((camera_resolution[1] / 2) - (bar_dimensions[1] / 2))
        ybar_end = ybar_start + bar_dimensions[1]

        y_length = ybar_end - ybar_start
        y_random = random.sample(range(ybar_start, ybar_end), y_length)
        time = float(space_metric) * np.sqrt(2) / float(bar_speed)  # dt between one coloum an another one
        time_off = float(space_metric) * bar_dimensions[0] / float(bar_speed)

        for run in range(camera_resolution[0] + camera_resolution[1]):
            if run < camera_resolution[0]:
                x = camera_resolution[0] - 1 - run
                y = camera_resolution[1] - 1
                while x < camera_resolution[0] and y >= 0:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x + 1
                    y = y - 1
            else:
                x = 0
                y = camera_resolution[0] + camera_resolution[1] - run - 1
                while x < camera_resolution[0] and y >= 0:
                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp + time_off
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                    x = x + 1
                    y = y - 1

            timestamp = timestamp + time
    return eventsON, eventsOFF, timestamp


def FakeStimuliAEDAT(AEDATresolution, step_stamp):
    hor_EventsON = [[[] for i in range(AEDATresolution[0])] for j in range(AEDATresolution[1])]
    # vert_EventsON = [[[] for i in range(AEDATresolution[0])] for j in range(AEDATresolution[1])]

    stamp = 1  # 1 milliseconds

    # Horizontal and vertical stimuli
    for x in range(0, AEDATresolution[0]):
        for y in range(0, AEDATresolution[1]):
            hor_EventsON[y][x] = stamp
        stamp = step_stamp + stamp
    return hor_EventsON


def real_data2events(data, ATIScamera_resolution, camera_resolution):
    eventsON = []
    eventsOFF = []
    events = []

    speed_x = []
    speed_y = []

    # shift to restart timestamps from zero
    time_shift = data[0][1] * 80 * pow(10, -6)  # ms

    for indx in range(0, len(data)):

        x_shift = ((ATIScamera_resolution[0] / 2) - (camera_resolution[0] / 2))
        y_shift = ((ATIScamera_resolution[1] / 2) - (camera_resolution[1] / 2))

        if (data[indx][3] > x_shift and data[indx][3] < (ATIScamera_resolution[0] - x_shift) and data[indx][
            4] > y_shift and data[indx][4] < (ATIScamera_resolution[1] - y_shift)):

            event = Event()
            if (data[indx][2] == 1):
                event.timestamp = data[indx][1] * 80 * pow(10, -6) - time_shift
                event.polarity = data[indx][2]
                event.x = data[indx][3] - x_shift
                event.y = data[indx][4] - y_shift

                events.append(event)
                eventsON.append(event)
            if (data[indx][2] == 0):
                event.timestamp = data[indx][1] * 80 * pow(10, -6) - time_shift
                event.polarity = data[indx][2]
                event.x = data[indx][3] - x_shift
                event.y = data[indx][4] - y_shift

                events.append(event)
                eventsOFF.append(event)

    time_window = 100  # ms
    old_timestamp = 0
    x = []
    y = []
    x_tmp = []
    y_tmp = []
    time = []
    speed_x_mean = []
    speed_y_mean = []
    for curr in range(0, len(events)):
        dt = np.float(abs(events[curr].timestamp - old_timestamp))
        x_tmp.append(events[curr].x)
        y_tmp.append(events[curr].y)

        if (dt > time_window):
            old_timestamp = events[curr].timestamp
            x.append(np.mean(x_tmp))
            y.append(np.mean(y_tmp))
            time.append(dt)

    for item in range(1, len(time)):
        speed_x_mean.append(abs(x[item] - x[item - 1]) / time[item])
        speed_y_mean.append(abs(y[item] - y[item - 1]) / time[item])
    speed_x_mean = np.mean(speed_x_mean)
    speed_y_mean = np.mean(speed_y_mean)
    return eventsON, eventsOFF, events, speed_x_mean, speed_y_mean


def FakestimuliAEDATformat(camera_resolution, deltaTcoloumn):
    hor_EventsON = FakeStimuliAEDAT(camera_resolution, deltaTcoloumn);
    entryHor = []
    # entryVer = []

    for x in range(camera_resolution[0]):
        for y in range(camera_resolution[1]):
            entryHor.append([hor_EventsON[x][y]])
            # entryVer.append([vert_EventsON[x][y]])
    return entryHor


def define_neuron_parameters(blob, cm, i_offset, tau_m, tau_refrac, tau_syn_E, tau_syn_EWTA, tau_syn_IWTA, tau_syn_I,
                             v_reset, v_rest, v_thresh):
    cell_params_semd = {'cm': cm,
                        'i_offset': i_offset,  # offset current
                        'tau_m': tau_m,  # membrane potential time constant	10 default
                        'tau_refrac': tau_refrac,  # refractory period time constant
                        'tau_syn_E': tau_syn_E,  # excitatory current time constant was 20
                        'tau_syn_I': blob,  # inhibitory current time constant was 20
                        'v_reset': v_reset,  # reset potential
                        'v_rest': v_rest,  # resting potential
                        'v_thresh': v_thresh  # spiking threshold
                        }

    cell_params_wta = {'cm': cm,
                       'i_offset': i_offset,  # offset current
                       'tau_m': tau_m,  # membrane potential time constant [ms]
                       'tau_refrac': tau_refrac,  # refractory period time constant [ms]
                       'tau_syn_E': tau_syn_EWTA,  # excitatory current time constant [ms]
                       'tau_syn_I': tau_syn_IWTA,  # inhibitory current time constant [ms]
                       'v_reset': v_reset,  # reset potential
                       'v_rest': v_rest,  # resting potential
                       'v_thresh': v_thresh  # spiking threshold
                       }

    return cell_params_semd, cell_params_wta


def load_neuronsATIS(path_file):
    with open(path_file, "rb") as f:
        a = pickle.load(f)
        print "loaded file correctly!"
    return a


def plotting_sEMDandCLEANING(plot_FLAG, Data, Sizes, simulation_plot, total_period, bar_speed, direction, stimulus,
                             cleaning_layerFLAG, length_neurons):
    sptc = Data.sptc_spikes.segments[0].spiketrains
    lr_spikes = Data.semd_lr_spikes.segments[0].spiketrains
    rl_spikes = Data.semd_rl_spikes.segments[0].spiketrains
    tb_spikes = Data.semd_tb_spikes.segments[0].spiketrains
    bt_spikes = Data.semd_bt_spikes.segments[0].spiketrains

    # PLOTTING RESULTS sEMD
    figsEMD = plt.figure()
    figsEMD.suptitle('Firing Rate, sEMD pop, tot neurons:, ' + str(length_neurons) + ', ' + stimulus + ' stim, ' + str(
        bar_speed) + ' [px/ms]', fontsize=16)

    ax1 = figsEMD.add_subplot(2, 2, 1)
    ax2 = figsEMD.add_subplot(2, 2, 2)
    ax3 = figsEMD.add_subplot(2, 2, 3)
    ax4 = figsEMD.add_subplot(2, 2, 4)

    ax1.set_ylabel('NeuronID')
    ax2.set_ylabel('NeuronID')
    ax3.set_ylabel('NeuronID')
    ax4.set_ylabel('NeuronID')

    ax1.set_xlabel('time (ms)')
    ax2.set_xlabel('time (ms)')
    ax3.set_xlabel('time (ms)')
    ax4.set_xlabel('time (ms)')

    spks_sptc = 0
    spks_lr = 0
    spks_rl = 0
    spks_tb = 0
    spks_bt = 0

    # RASTER PLOT sEMD
    for i in range(Sizes.pop_semd):
        # ax5.plot(Data.sptc_spikes.segments[0].spiketrains[i], [i] * len(Data.sptc_spikes.segments[0].spiketrains[i]),".b")
        # spks_sptc = spks_sptc	+len(Data.sptc_spikes.segments[0].spiketrains[i])
        spks_lr = spks_lr + len(Data.semd_lr_spikes.segments[0].spiketrains[i])
        spks_rl = spks_rl + len(Data.semd_rl_spikes.segments[0].spiketrains[i])
        spks_tb = spks_tb + len(Data.semd_tb_spikes.segments[0].spiketrains[i])
        spks_bt = spks_bt + len(Data.semd_bt_spikes.segments[0].spiketrains[i])

        if (plot_FLAG):
            # ax1_1.plot(Data.sptc_spikes.segments[0].spiketrains[i], [i] * len(Data.sptc_spikes.segments[0].spiketrains[i]), ".g", ms = 0.1)
            ax1.plot(Data.semd_lr_spikes.segments[0].spiketrains[i], [i] * \
                     (len(Data.semd_lr_spikes.segments[0].spiketrains[i])), ".r")
            ax2.plot(Data.semd_rl_spikes.segments[0].spiketrains[i], [i] * \
                     (len(Data.semd_rl_spikes.segments[0].spiketrains[i])), ".g")
            ax3.plot(Data.semd_tb_spikes.segments[0].spiketrains[i], [i] * \
                     (len(Data.semd_tb_spikes.segments[0].spiketrains[i])), ".b")
            ax4.plot(Data.semd_bt_spikes.segments[0].spiketrains[i], [i] * \
                     (len(Data.semd_bt_spikes.segments[0].spiketrains[i])), ".m")

    spks_lr = spks_lr / float(total_period) * 1000.0
    spks_rl = spks_rl / float(total_period) * 1000.0
    spks_tb = spks_tb / float(total_period) * 1000.0
    spks_bt = spks_bt / float(total_period) * 1000.0

    ax1.set_xlim([0, simulation_plot])
    ax2.set_xlim([0, simulation_plot])
    ax3.set_xlim([0, simulation_plot])
    ax4.set_xlim([0, simulation_plot])

    ax1.set_title('Mean Firing Rate LR sEMD population [Hz]: %i' % spks_lr)
    ax2.set_title('Mean Firing Rate RL sEMD population [Hz]: %i' % spks_rl)
    ax3.set_title('Mean Firing Rate TB sEMD population [Hz]: %i' % spks_tb)
    ax4.set_title('Mean Firing Rate BT sEMD population [Hz]: %i' % spks_bt)

    string_bar_speed = str(bar_speed)
    string_bar_speed = re.sub(r'[^\w\s]', '', string_bar_speed)
    if (plot_FLAG):
        figsEMD.savefig(string_bar_speed + 'sEMD' + direction)

    # PLOTTING RESULTS CLEANING LAYER
    fig_cleaning = plt.figure()
    fig_cleaning.suptitle(
        'Firing Rate, cleaning pop, tot neurons:, ' + str(length_neurons) + ', ' + stimulus + ' stim, ' + str(
            bar_speed) + ' [px/ms]', fontsize=16)
    ax1 = fig_cleaning.add_subplot(2, 2, 1)
    ax2 = fig_cleaning.add_subplot(2, 2, 2)
    ax3 = fig_cleaning.add_subplot(2, 2, 3)
    ax4 = fig_cleaning.add_subplot(2, 2, 4)

    ax3.set_xlabel('time (ms)')
    ax4.set_xlabel('time (ms)')
    ax1.set_ylabel('NeuronID')
    ax3.set_ylabel('NeuronID')

    spks_sptc = 0
    spks_cleaning_lr = 0
    spks_cleaning_rl = 0
    spks_cleaning_tb = 0
    spks_cleaning_bt = 0

    if (cleaning_layerFLAG):

        # RASTER PLOT CLEANING
        for i in range(Sizes.pop_semd):
            # ax5.plot(Data.sptc_spikes.segments[0].spiketrains[i], [i] * len(Data.sptc_spikes.segments[0].spiketrains[i]),".b")
            # spks_sptc = spks_sptc	+len(Data.sptc_spikes.segments[0].spiketrains[i])
            spks_cleaning_lr = spks_cleaning_lr + len(Data.cleaning_layer_lr_spikes.segments[0].spiketrains[i])
            spks_cleaning_rl = spks_cleaning_rl + len(Data.cleaning_layer_rl_spikes.segments[0].spiketrains[i])
            spks_cleaning_tb = spks_cleaning_tb + len(Data.cleaning_layer_tb_spikes.segments[0].spiketrains[i])
            spks_cleaning_bt = spks_cleaning_bt + len(Data.cleaning_layer_bt_spikes.segments[0].spiketrains[i])
            if (plot_FLAG):
                # ax1_1.plot(Data.sptc_spikes.segments[0].spiketrains[i], [i] * len(Data.sptc_spikes.segments[0].spiketrains[i]), ".g", ms = 0.1)
                ax1.plot(Data.cleaning_layer_lr_spikes.segments[0].spiketrains[i], [i] * \
                         (len(Data.cleaning_layer_lr_spikes.segments[0].spiketrains[i])), ".r")
                ax2.plot(Data.cleaning_layer_rl_spikes.segments[0].spiketrains[i], [i] * \
                         (len(Data.cleaning_layer_rl_spikes.segments[0].spiketrains[i])), ".g")
                ax3.plot(Data.cleaning_layer_tb_spikes.segments[0].spiketrains[i], [i] * \
                         (len(Data.cleaning_layer_tb_spikes.segments[0].spiketrains[i])), ".b")
                ax4.plot(Data.cleaning_layer_bt_spikes.segments[0].spiketrains[i] / total_period, [i] * \
                         len(Data.cleaning_layer_bt_spikes.segments[0].spiketrains[i]), ".m")

        spks_cleaning_lr = spks_cleaning_lr / float(total_period) * 1000.0
        spks_cleaning_rl = spks_cleaning_rl / float(total_period) * 1000.0
        spks_cleaning_tb = spks_cleaning_tb / float(total_period) * 1000.0
        spks_cleaning_bt = spks_cleaning_bt / float(total_period) * 1000.0

        ax1.set_xlim([0, simulation_plot])
        ax2.set_xlim([0, simulation_plot])
        ax3.set_xlim([0, simulation_plot])
        ax4.set_xlim([0, simulation_plot])
        ax1.set_title('Firing Rate of spikes for L2R [Hz]: %i' % spks_cleaning_lr)
        ax2.set_title('Firing Rate of spikes for R2L [Hz]: %i' % spks_cleaning_rl)
        ax3.set_title('Firing Rate of spikes for T2B [Hz]: %i' % spks_cleaning_tb)
        ax4.set_title('Firing Rate of spikes for B2T [Hz]: %i' % spks_cleaning_bt)

        if (plot_FLAG):
            fig_cleaning.savefig(string_bar_speed + 'cleaning' + direction)

    return spks_lr, spks_rl, spks_tb, spks_bt, spks_cleaning_lr, spks_cleaning_rl, spks_cleaning_tb, spks_cleaning_bt


def saving_data_sEMDandCLEANING(saving_data_path, Data, Sizes, total_period, direction, cleaning_layerFLAG):
    Data_path_file = saving_data_path + 'Data' + direction + '.pickle'
    with open(Data_path_file, "wb") as f:
        pickle.dump(Data, f)

    spks_lr = []
    spks_rl = []
    spks_tb = []
    spks_bt = []

    # RASTER PLOT sEMD
    for i in range(Sizes.pop_semd):
        # ax5.plot(Data.sptc_spikes.segments[0].spiketrains[i], [i] * len(Data.sptc_spikes.segments[0].spiketrains[i]),".b")
        # spks_sptc = spks_sptc	+len(Data.sptc_spikes.segments[0].spiketrains[i])
        spks_lr.append(float(len(Data.semd_lr_spikes.segments[0].spiketrains[i])) / total_period * 1000)
        spks_rl.append(float(len(Data.semd_rl_spikes.segments[0].spiketrains[i])) / total_period * 1000)
        spks_tb.append(float(len(Data.semd_tb_spikes.segments[0].spiketrains[i])) / total_period * 1000)
        spks_bt.append(float(len(Data.semd_bt_spikes.segments[0].spiketrains[i])) / total_period * 1000)

    spks_cleaning_lr = []
    spks_cleaning_rl = []
    spks_cleaning_tb = []
    spks_cleaning_bt = []

    if (cleaning_layerFLAG):

        # RASTER PLOT CLEANING
        for i in range(Sizes.pop_semd):
            # ax5.plot(Data.sptc_spikes.segments[0].spiketrains[i], [i] * len(Data.sptc_spikes.segments[0].spiketrains[i]),".b")
            # spks_sptc = spks_sptc	+len(Data.sptc_spikes.segments[0].spiketrains[i])
            spks_cleaning_lr.append(
                float(len(Data.cleaning_layer_lr_spikes.segments[0].spiketrains[i])) / total_period * 1000)
            spks_cleaning_rl.append(
                float(len(Data.cleaning_layer_rl_spikes.segments[0].spiketrains[i])) / total_period * 1000)
            spks_cleaning_tb.append(
                float(len(Data.cleaning_layer_tb_spikes.segments[0].spiketrains[i])) / total_period * 1000)
            spks_cleaning_bt.append(
                float(len(Data.cleaning_layer_bt_spikes.segments[0].spiketrains[i])) / total_period * 1000)

    return spks_lr, spks_rl, spks_tb, spks_bt, spks_cleaning_lr, spks_cleaning_rl, spks_cleaning_tb, spks_cleaning_bt


def ATISformat2neuron(camera_resolution, data):
    RFs_events = []
    counterRFs = 0
    for y in range(0, camera_resolution[1]):
        for x in range(0, camera_resolution[0]):
            rf = RF()
            rf.center_x = x
            rf.center_y = y
            rf.RF_id = counterRFs
            counterRFs = counterRFs + 1
            RFs_events.append(rf)

    neurons = [[] for _ in range(len(RFs_events))]
    for e in tqdm(data):
        for rf in RFs_events:
            if (rf.center_x == e.x and rf.center_y == e.y):
                neurons[rf.RF_id].append(e.timestamp)
    return neurons
