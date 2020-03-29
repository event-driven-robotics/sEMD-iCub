
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 23:19:18 2019

@author: giulia.dangelo@iit.it
"""

from EccentricityFuncs import *
from decode_events_functions import*
import spynnaker8 as p
import sys
import numpy as np
from EccentricitysEMDFunctions import *
from EccentricityFuncs import *
import matplotlib.pyplot as plt
from sEMDFunctions import *
from aedat_spike_converter import *

# IMPORTED BUT NOT USED
#from spynnaker.pyNN.models.neuron.builds import IFCurrExpSEMD
#import struct
#import matplotlib as mpl
#from pyNN.utility.plotting import Figure, Panel
#import os
#import random

cleaning_layerFLAG = True
AEDATrealdataFLAG=False
ATISrealdataFLAG=True
ATISfakestimuliFLAG=False

multiple_speedsFLAG = False

if (AEDATrealdataFLAG):
    print 'AEDAT real data'

    stimulus = 'LR'
    # stimulus = 'RL'
    # stimulus = 'BT'
    # stimulus = 'TB'

    camera_resolution = [128, 128]
    #path_input = '/home/giulia/workspace/tde-iCubIIT/Stimuli/ThoerbenStimuli/icub_semd/recordings1/pattern_2_light_6_speed_3_number_2.aedat'
    path_input = '/home/giulia/workspace/tde-iCubIIT/Stimuli/ThoerbenStimuli/recordings2/pattern_2_light_6_speed_3_number_3.aedat'
    N_AER_ADRESSES = 128**2*2
    BAR_SPEED=[0]
    # convert aedat file into spike array
    [record_on, record_off ]= aedat_to_spikes(path_input, N_AER_ADRESSES)
    eventsDAVIS=reorderDAVISdata(record_on, camera_resolution)

    plot_FLAG = True
    events = eventsDAVIS

if (ATISrealdataFLAG):
    print 'ATIS real data'

    #ATIS parameters
    ATIScamera_resolution=[304, 240]
    camera_resolution = [160, 160]
    plot_FLAG = True

    stimulus = 'LR'
    # stimulus = 'RL'
    # stimulus = 'BT'
    # stimulus = 'TB'

    #dataset_path= '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/circle_moving/LR/data.log.txt'
    #dataset_path= '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/bar_moving/LR/data.log.txt'

    dataset_path ='/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving_speed/LR5/data.log.txt'
    # dataset_path = '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving/RL/data.log.txt'
    #dataset_path = '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving/TB/data.log.txt'
    #dataset_path = '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving/BT/data.log.txt'

    [left_data, right_data]=ATISload_data(ATIScamera_resolution, dataset_path)
    # e(channel, timestamp, polarity, x, y)
    [eventsON, eventsOFF, eventsTOT, speed_x_mean, speed_y_mean]=real_data2events(right_data, ATIScamera_resolution, camera_resolution)
    neurons_AEDATformat=ATISformat2neuron(camera_resolution, eventsON)  #conversion from events to neurond_ID events
    events=neurons_AEDATformat

    if stimulus=='LR' or stimulus=='RL':
        BAR_SPEED = [speed_x_mean]
    elif stimulus=='TB' or stimulus=='BT':
        BAR_SPEED = [speed_y_mean]

    total_period = eventsON[-1].timestamp - eventsON[0].timestamp

    file_location= '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/VisualisedStimuli'
    file_name= 'stimulus'

    #direction
    split_dataset_path=dataset_path.split("/")
    direction=split_dataset_path[8]

    frame_rate=5
    #create_video(file_location, file_name, frame_rate, eventsON, camera_resolution[0], camera_resolution[1], direction)

    saving_data_path  = '/home/giulia/workspace/tde-iCubIIT/Code/sEMD/results/sEMD/ResultsATISdatasets/LR5weights03clean02delayfac01cam160/'

if (ATISfakestimuliFLAG):
    print 'ATIS fake stimuli creation to be fed into the sEMD'

    saving_data_path = '/home/giulia/workspace/tde-iCubIIT/Code/sEMD/results/EccentricitysEMD/FakeStimuli/PAPER/sEMD/10weights03clean02delayfac1cam160/'

    stimulus = 'LR'
    # stimulus = 'RL'
    # stimulus = 'BT'
    # stimulus = 'TB'

    if(multiple_speedsFLAG):
        BAR_SPEED=[]

        start_speed = 0.01
        end_speed = 1.0
        for speed in np.arange(start_speed, end_speed, 0.03):
            curr_speed=round(speed, 2)
            BAR_SPEED.append(curr_speed)

        plot_FLAG = False
    else:
        BAR_SPEED = [1.0]
        plot_FLAG = True

    space_metric = 1  # pixel
    camera_resolution = [160, 160]
    bar_dimensions = [50, 160]




LR = []
RL = []
TB = []
BT = []

CLEANING_LR = []
CLEANING_RL = []
CLEANING_TB = []
CLEANING_BT = []

LR_data = []
RL_data = []
TB_data = []
BT_data = []

CLEANING_LR_data = []
CLEANING_RL_data = []
CLEANING_TB_data = []
CLEANING_BT_data = []


for speed_indx in range(0, len(BAR_SPEED)):

    bar_speed=BAR_SPEED[speed_indx]

    #contrast='BlackOverWhite'
    contrast='WhiteOverBlack'

    if (ATISfakestimuliFLAG):

        direction = 'LR'
        # direction='RL'
        # direction='BT'
        # direction='TB'

        [eventsATIS_ON, eventsATIS_OFF, total_period]=FakeStimuliBarMoving(camera_resolution, bar_speed, space_metric, bar_dimensions, direction, contrast)
        neurons_AEDATformat = ATISformat2neuron(camera_resolution,eventsATIS_ON)  # conversion from events to neurond_ID events

        if stimulus == 'LR' or stimulus == 'RL':
            total_space = camera_resolution[0]  # mm (30 cm)
        elif stimulus == 'TB' or stimulus == 'BT':
            total_space = camera_resolution[1]  # mm (30 cm)

        events = neurons_AEDATformat

    'sEMD'
    entryHor=events
    inputs =[{'spike_times': entryHor}]
    deltaTcoloumn= 7

    neurons_row = camera_resolution[0]/4
    length_neurons = neurons_row** 2

    class Sizes:
        pass

        edge_dvs = camera_resolution[0]
        edge_sptc = neurons_row
        edge_semd = neurons_row
        pop_dvs = edge_dvs ** 2 * 2
        pop_sptc = edge_sptc ** 2
        pop_semd = edge_semd ** 2

    class Weights:
        pass

        sptc = 0.3  # was 0.6
        semd = 0.3  # was 0.5
        cleaning = 0.2

    # NEURON PARAMETERS
    blob = 20
    cm = 0.25
    i_offset = 0
    tau_m = 10
    tau_refrac = 1
    tau_syn_E = 20
    tau_syn_I = 20
    v_reset = -85
    v_rest = -60
    v_thresh = -50
    tau_syn_EWTA=100
    tau_syn_IWTA=300

    [cell_params_semd, cell_params_sptc] = define_neuron_parameters(blob, cm, i_offset, tau_m, tau_refrac, tau_syn_E, tau_syn_EWTA, tau_syn_IWTA,  tau_syn_I, v_reset, v_rest, v_thresh)

    # NEURON POPULATION, Loop to simulate different stimuli consecutively
    for i in inputs:
        ## NEURONS
        delay = 1

        ## RECORDED INPUT
        maxTimestamp = 0
        minTimestamp = sys.maxsize
        aedatLength = 0
        endTime = 0

        # SET UP SIMULATION
        simulation_timestep = 1  # ms
        simulation_runtime = total_period + 500  # total period + buffer time
        simulation_plot = total_period + 200
        p.setup(timestep=simulation_timestep)
        p.set_number_of_neurons_per_core(p.extra_models.IF_curr_exp_sEMD, 50)  # it was 100


        class Populations:
            pass
            dvs = p.Population(camera_resolution[0]**2, p.SpikeSourceArray, i, label='Input')
            sptc = p.Population(Sizes.pop_sptc, p.IF_curr_exp, cell_params_sptc, label="SPTC")
            semd_lr = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label="sEMD lr")
            semd_rl = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label="sEMD rl")
            semd_tb = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label="sEMD tb")
            semd_bt = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label="sEMD bt")
            cleaning_layer_lr = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label="sEMD lr")
            cleaning_layer_rl = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label="sEMD rl")
            cleaning_layer_tb = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label="sEMD tb")
            cleaning_layer_bt = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label="sEMD bt")



        # connection matrix

        # prepare complex connection matrix between dvs and sptc
        scaling_factor = Sizes.edge_dvs / Sizes.edge_sptc
        sptc = np.arange(Sizes.pop_sptc)
        sptc.resize(Sizes.edge_sptc, Sizes.edge_sptc)
        sptc = np.rot90(sptc, 0)
        sptc = np.kron(sptc, np.ones((int(scaling_factor), int(scaling_factor))))
        sptc.resize(Sizes.pop_dvs / 2)
        hood = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                hood.append(-j + i * neurons_row)
        neighbours = []

        LRremove = range((neurons_row-1), neurons_row**2, neurons_row)
        RLremove = range(0, neurons_row**2, neurons_row)


        delay_fac=1
        delay_cleaning = 0
        class Connect:
            pass
            sptc = [(i, sptc[i], Weights.sptc, delay) for i in range(Sizes.pop_dvs / 2)]

            #LR
            semd_lr_fac = [(i, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd)]
            semd_lr_fac = np.delete(semd_lr_fac, LRremove, 0)
            semd_lr_trig = [(i + 1, i, Weights.semd, delay) for i in range(Sizes.pop_semd)]
            semd_lr_trig = np.delete(semd_lr_trig, LRremove, 0)

            #RL
            semd_rl_fac = [(i + 1, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd)]
            semd_rl_fac = np.delete(semd_rl_fac, RLremove, 0)
            semd_rl_trig = [(i, i, Weights.semd, delay) for i in range(Sizes.pop_semd)]
            semd_rl_trig = np.delete(semd_rl_trig, RLremove, 0)

            #TB
            semd_tb_fac = [(i, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd-neurons_row)]
            semd_tb_trig = [(i + neurons_row, i, Weights.semd, delay) for i in range(Sizes.pop_semd-neurons_row)]

            #BT
            semd_bt_fac = [(i + neurons_row, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd-neurons_row)]
            semd_bt_trig = [(i, i, Weights.semd, delay) for i in range(Sizes.pop_semd-32)]

            cleaning = [(i, i, Weights.cleaning , delay) for i in range(Sizes.pop_semd)]


        # projections
        p.Projection(Populations.dvs, Populations.sptc, p.FromListConnector(Connect.sptc), \
                     p.StaticSynapse(), receptor_type='excitatory')
        p.Projection(Populations.sptc, Populations.semd_lr, p.FromListConnector(Connect.semd_lr_fac) \
                     , p.StaticSynapse(), receptor_type='excitatory')
        p.Projection(Populations.sptc, Populations.semd_lr, p.FromListConnector(Connect.semd_lr_trig) \
                     , p.StaticSynapse(), receptor_type='excitatory2')
        p.Projection(Populations.sptc, Populations.semd_rl, p.FromListConnector(Connect.semd_rl_fac) \
                     , p.StaticSynapse(), receptor_type='excitatory')
        p.Projection(Populations.sptc, Populations.semd_rl, p.FromListConnector(Connect.semd_rl_trig) \
                     , p.StaticSynapse(), receptor_type='excitatory2')
        p.Projection(Populations.sptc, Populations.semd_tb, p.FromListConnector(Connect.semd_tb_fac) \
                     , p.StaticSynapse(), receptor_type='excitatory')
        p.Projection(Populations.sptc, Populations.semd_tb, p.FromListConnector(Connect.semd_tb_trig) \
                     , p.StaticSynapse(), receptor_type='excitatory2')
        p.Projection(Populations.sptc, Populations.semd_bt, p.FromListConnector(Connect.semd_bt_fac) \
                     , p.StaticSynapse(), receptor_type='excitatory')
        p.Projection(Populations.sptc, Populations.semd_bt, p.FromListConnector(Connect.semd_bt_trig) \
                     , p.StaticSynapse(), receptor_type='excitatory2')
        # for i in neighbours:
        #     p.Projection(Populations.semd_lr, Populations.semd_rl, p.FromListConnector(i) \
        #                  , p.StaticSynapse(), receptor_type='inhibitory')
        #     p.Projection(Populations.semd_rl, Populations.semd_lr, p.FromListConnector(i) \
        #                  , p.StaticSynapse(), receptor_type='inhibitory')
        #     p.Projection(Populations.semd_bt, Populations.semd_tb, p.FromListConnector(i) \
        #                  , p.StaticSynapse(), receptor_type='inhibitory')
        #     p.Projection(Populations.semd_tb, Populations.semd_bt, p.FromListConnector(i) \
        #                  , p.StaticSynapse(), receptor_type='inhibitory')
        # inhibitory projections

        p.Projection(Populations.semd_lr, Populations.cleaning_layer_rl, p.FromListConnector(Connect.cleaning), p.StaticSynapse(), receptor_type= 'inhibitory')
        p.Projection(Populations.semd_rl, Populations.cleaning_layer_lr, p.FromListConnector(Connect.cleaning),  p.StaticSynapse(),receptor_type= 'inhibitory')
        p.Projection(Populations.semd_tb, Populations.cleaning_layer_bt, p.FromListConnector(Connect.cleaning), p.StaticSynapse(), receptor_type= 'inhibitory')
        p.Projection(Populations.semd_bt, Populations.cleaning_layer_tb, p.FromListConnector(Connect.cleaning), p.StaticSynapse(), receptor_type= 'inhibitory')

        p.Projection(Populations.semd_rl, Populations.cleaning_layer_rl, p.FromListConnector(Connect.cleaning), p.StaticSynapse(), receptor_type= 'excitatory')
        p.Projection(Populations.semd_lr, Populations.cleaning_layer_lr, p.FromListConnector(Connect.cleaning), p.StaticSynapse(), receptor_type= 'excitatory')
        p.Projection(Populations.semd_bt, Populations.cleaning_layer_bt, p.FromListConnector(Connect.cleaning), p.StaticSynapse(), receptor_type= 'excitatory')
        p.Projection(Populations.semd_tb, Populations.cleaning_layer_tb, p.FromListConnector(Connect.cleaning), p.StaticSynapse(), receptor_type= 'excitatory')




        # records
        Populations.sptc.record(['spikes'])
        Populations.semd_lr.record(['spikes'])  # ,'gsyn_exc','gsyn_inh'])
        Populations.semd_rl.record(['spikes'])  # ,'gsyn_exc','gsyn_inh'])
        Populations.semd_tb.record(['spikes'])  # ,'gsyn_exc','gsyn_inh'])
        Populations.semd_bt.record(['spikes'])  # ,'gsyn_exc','gsyn_inh'])
        Populations.cleaning_layer_bt.record(({'spikes'}))
        Populations.cleaning_layer_tb.record(({'spikes'}))
        Populations.cleaning_layer_rl.record(({'spikes'}))
        Populations.cleaning_layer_lr.record(({'spikes'}))

        # run simulation
        p.run(simulation_runtime)


        class Data:
            pass

            # receive data from neurons
            sptc_spikes = Populations.sptc.get_data(['spikes'])
            semd_lr_spikes = Populations.semd_lr.get_data(['spikes'])
            semd_rl_spikes = Populations.semd_rl.get_data(['spikes'])
            semd_bt_spikes = Populations.semd_bt.get_data(['spikes'])
            semd_tb_spikes = Populations.semd_tb.get_data(['spikes'])
            cleaning_layer_lr_spikes = Populations.cleaning_layer_lr.get_data(['spikes'])
            cleaning_layer_rl_spikes = Populations.cleaning_layer_rl.get_data(['spikes'])
            cleaning_layer_tb_spikes = Populations.cleaning_layer_tb.get_data(['spikes'])
            cleaning_layer_bt_spikes = Populations.cleaning_layer_bt.get_data(['spikes'])


        if (multiple_speedsFLAG):
            print 'multiple speed processing for speed ' + str(bar_speed)
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

            spks_lr = spks_lr / float(total_period) * 1000.0
            spks_rl = spks_rl / float(total_period) * 1000.0
            spks_tb = spks_tb / float(total_period) * 1000.0
            spks_bt = spks_bt / float(total_period) * 1000.0

            # data from all the population
            LR.append(spks_lr)
            RL.append(spks_rl)
            TB.append(spks_tb)
            BT.append(spks_bt)

        else:

            print 'saving all data '

            [data_spks_lr, data_spks_rl, data_spks_tb, data_spks_bt, data_spks_cleaning_lr, data_spks_cleaning_rl, data_spks_cleaning_tb, data_spks_cleaning_bt]=saving_data_sEMDandCLEANING(saving_data_path, Data, Sizes, total_period, direction, cleaning_layerFLAG)
            [spks_lr, spks_rl, spks_tb, spks_bt, spks_cleaning_lr, spks_cleaning_rl, spks_cleaning_tb, spks_cleaning_bt]=plotting_sEMDandCLEANING(plot_FLAG, Data, Sizes, simulation_plot, total_period, bar_speed, direction, stimulus, cleaning_layerFLAG, length_neurons)

            #data from all the population
            LR.append(spks_lr)
            RL.append(spks_rl)
            TB.append(spks_tb)
            BT.append(spks_bt)

            CLEANING_LR.append(spks_cleaning_lr)
            CLEANING_RL.append(spks_cleaning_rl)
            CLEANING_TB.append(spks_cleaning_tb)
            CLEANING_BT.append(spks_cleaning_bt)

        p.end()

    if multiple_speedsFLAG:
        name_file = 'LR'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(LR, f)

        name_file = 'RL'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(RL, f)

        name_file = 'TB'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(TB, f)

        name_file = 'BT'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(BT, f)
    else:

        name_file = 'LR'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(LR, f)

        name_file = 'RL'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(RL, f)

        name_file = 'TB'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(TB, f)

        name_file = 'BT'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(BT, f)


        name_file='CLEANING_LR'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(CLEANING_LR, f)

        name_file='CLEANING_RL'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(CLEANING_RL, f)

        name_file='CLEANING_TB'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(CLEANING_TB, f)

        name_file='CLEANING_BT'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(CLEANING_BT, f)

        name_file='BAR_SPEED'
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(BAR_SPEED, f)


    # fig1 = plt.figure()
    # plt.plot(BAR_SPEED, CLEANING_LR, 'r--', BAR_SPEED, CLEANING_RL, 'b--', BAR_SPEED, CLEANING_TB, 'g--',  CLEANING_BT, 'm--')
    # plt.xlabel('bar speed [px/ms]')
    # plt.ylabel('Firing Rate [Hz]')
    # fig1.savefig('Cleaning Layer Neurons Firing Rate vs Bar Speed')

    if (plot_FLAG):
        plt.show()