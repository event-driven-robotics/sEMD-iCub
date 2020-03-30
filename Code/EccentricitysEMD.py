#!/usr/bin/env python20
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:32:49 2019

@author: giulia.dangelo@iit.it
"""
import matplotlib
matplotlib.use('Agg')

from EccentricityFuncs import *
from decode_events_functions import*
import spynnaker8 as p
from aedat_spike_converter import *
import sys
import numpy as np
from EccentricitysEMDFunctions import *
import matplotlib.pyplot as plt


# IMPORTED BUT NOT USED

#from spynnaker.pyNN.models.neuron.builds import IFCurrExpSEMD
#import struct
#import matplotlib as mpl
#from pyNN.utility.plotting import Figure, Panel
#import os
#import random

wtaL = False
if(wtaL):
    cleaning_layerFLAG = False
else:
    cleaning_layerFLAG = True



multiple_speedsFLAG=False

AEDATrealdataFLAG=False
ATISrealdataFLAG=False
ATISfakestimuliFLAG=True


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

    saving_data_path = '/home/giulia/workspace/tde-iCubIIT/Code/sEMD/results/EccentricitysEMD/ResultsATISdatasets/LR1weights03clean02delayfac01cam160/'

    #ATIS parameters
    ATIScamera_resolution=[304, 240]
    camera_resolution = [160, 160]
    plot_FLAG = True

    stimulus = 'LR'
    # stimulus = 'RL'
    # stimulus = 'BT'
    # stimulus = 'TB'


    dataset_path = '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving_speed/LR1/data.log.txt'


    #dataset_path = '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving/RL/data.log.txt'
    #dataset_path = '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving/TB/data.log.txt'
    #dataset_path = '/home/giulia/workspace/tde-iCubIIT/Stimuli/semd_Datasets/line_moving/BT/data.log.txt'

    [left_data, right_data]=ATISload_data(ATIScamera_resolution, dataset_path)
    # e(channel, timestamp, polarity, x, y)
    [eventsON, eventsOFF, eventsTOT, speed_x_mean, speed_y_mean]=real_data2events(right_data, ATIScamera_resolution, camera_resolution)
    # neurons_AEDATformat=ATISformat2neuron(camera_resolution, eventsON)
    events = eventsON

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
    # create_video(file_location, file_name, frame_rate, eventsON, camera_resolution[0], camera_resolution[1], direction)



if (ATISfakestimuliFLAG):
    print 'ATIS fake stimuli creation to be fed into the sEMD'

    saving_data_path = '/localhome/mbaxrap7/giulia/03weights03clean02delayfac1cam160LRbar50/'
    # saving_data_path = '/home/giulia/Desktop/prova/'
    # saving_data_path = '/home/giulia/workspace/tde-iCubIIT/Code/sEMD/results/EccentricitysEMD/FakeStimuli/multiple_speedsECCsEMD/'

    stimulus = 'LR'
    # stimulus = 'RL'
    # stimulus = 'BT'
    # stimulus = 'TB'
    # stimulus = 'TransTlBr'
    stimulus = 'TransTrBl'
    stimulus = 'TransBlTr'
    stimulus = 'TransBrTl'


    direction = stimulus
    if(multiple_speedsFLAG):
        BAR_SPEED=[]

        start_speed = 0.01
        end_speed = 1.0
        for speed in np.arange(start_speed, end_speed, 0.03):
            curr_speed=round(speed, 2)
            BAR_SPEED.append(curr_speed)

        plot_FLAG = False
    else:
        BAR_SPEED = [0.3]
        plot_FLAG = True

    space_metric = 1  # pixel
    camera_resolution = [160, 160]
    bar_dimensions = [50, 50]



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


        [eventsATIS_ON, eventsATIS_OFF, total_period]=FakeStimuliBarMoving(camera_resolution, bar_speed, space_metric, bar_dimensions, direction, contrast)
        # if stimulus == 'LR' or stimulus == 'RL':
        #     total_space = camera_resolution[0]  # mm (30 cm)
        # elif stimulus == 'TB' or stimulus == 'BT':
        #     total_space = camera_resolution[1]  # mm (30 cm)

        events = eventsATIS_ON

    'Eccentricity layer'

    # RFs LAYER
    maximum_kernel_size = 10
    RFthreshold_percentage = 60
    fovea_size_percentage = 10
    fovea_dimensions = fovea_percentage(fovea_size_percentage, camera_resolution)
    tau=float(1000)

    # createRFsFLAG = False
    if speed_indx>0:
        createRFsFLAG=False
    else:
        createRFsFLAG = True



    if (createRFsFLAG):
        RFs = RFsEccentricity(maximum_kernel_size, camera_resolution, fovea_dimensions, RFthreshold_percentage, tau)

        'Eccentricity processing'

        # DISPLAY ECCENTRICITY LAYER + NEURONS LOOK UP TABLE
        folder_checking_eccentricity = 'CheckingEccentricity/'
        folder_checking_trajectories = folder_checking_eccentricity + 'trajectories/'
        [coordinates_x, coordinates_y, y_trajectories] = display_eccentricity_rings_coordinates(RFs, camera_resolution, folder_checking_eccentricity, saving_data_path)
        # trajectories_neuron_id(RFs, y_trajectories, camera_resolution, folder_checking_eccentricity)
        [RFs, counter_x, counter_y] = has_connection_xy_trajectories(RFs, y_trajectories, camera_resolution, saving_data_path)
        visualise_neuronsFLAG=False # to visualise each neuron jpeg
        [RFs, length_neurons] = has_connection_trajectories(RFs, camera_resolution, counter_x, counter_y, visualise_neuronsFLAG)

        # saving RFs
        RFs_path_file = saving_data_path + 'RFs_neurons_reordered'+str(camera_resolution[0])+'.pickle'
        with open(RFs_path_file, "wb") as f:
            pickle.dump(RFs, f)
    else:
        RFs_path_file = saving_data_path + 'RFs_neurons_reordered'+str(camera_resolution[0])+'.pickle'
        RFs = load_file(RFs_path_file)

        counter_x_path_file = saving_data_path + 'counter_x.pickle'
        counter_x = load_file(counter_x_path_file)

        counter_y_path_file = saving_data_path + 'counter_y.pickle'
        counter_y = load_file(counter_y_path_file)

        length_neurons = counter_x*counter_y

    'Eccentricity results'

    #PROCESSING DATA WITH ECCENTRICITY
    path_ToSavePickle= saving_data_path
    [RFs, spikes, neurons, path_file]=RFcell(events, RFs, path_ToSavePickle, tau,length_neurons)

    #WRITE TXT FILE BASED ON NEURON_ID
    nameTXTfile= "neurons"
    write_neurons_file(neurons, nameTXTfile, path_ToSavePickle)

    #LOAD NEURONS
    entryHor=load_file(path_file)
    'sEMD'
    inputs ={'spike_times': entryHor}
    deltaTcoloumn= 7


    class Sizes:
        pass

        edge_dvs = camera_resolution[0]
        #pop_dvs = length_neurons
        #pop_sptc = length_neurons
        pop_semd = length_neurons
        if (wtaL):
            edge_WTALocal = 15
            pop_WTA_Local=(int(np.sqrt(length_neurons))/edge_WTALocal)**2 #pop_semd/neurons_around

    class Weights:
        pass

        #sptc = 0.5  # previously 0.6
        semd = 0.3  # 0.1
        if (cleaning_layerFLAG):
            cleaning = 0.2
        if (wtaL):
            WTA_EXT = 0.5
            WTA_Inh = 0.8

    # NEURON ADDRESSES
    N_AER_ADRESSES = Sizes.edge_dvs**2*2


    # NEURON PARAMETERS
    blob = 20
    cm=0.25
    i_offset=0
    tau_m=10
    tau_refrac=1
    tau_syn_E=20
    tau_syn_I=20
    v_reset=-85
    v_rest=-60
    v_thresh=-50
    tau_syn_EWTA=100
    tau_syn_IWTA=300

    [cell_params_semd, cell_params_wta]=define_neuron_parameters(blob, cm, i_offset, tau_m, tau_refrac, tau_syn_E, tau_syn_EWTA, tau_syn_IWTA,  tau_syn_I, v_reset, v_rest, v_thresh)


    # NEURON POPULATION, Loop to simulate different stimuli consecutively
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
    buffer=20
    if (wtaL):
        num_cores=int((length_neurons+(length_neurons*4)+(length_neurons*4)+(Sizes.pop_WTA_Local))/864)+buffer  #864 48 chips, *18 cores
    else:
        num_cores=int((length_neurons+(length_neurons*4)+(length_neurons*4))/864)+buffer #864 48 chips, *18 cores

    p.set_number_of_neurons_per_core(p.extra_models.IF_curr_exp_sEMD, num_cores)

    class Populations:
        pass
        #dvs =  p.Population(AEDATresolution[0]**2, p.SpikeSourceArray, i, label='Input')
        dvs = p.Population(length_neurons, p.SpikeSourceArray, inputs, label='Input')
        # sptc = p.Population(Sizes.pop_sptc, p.IF_curr_exp, cell_params_sptc, label = "SPTC")
        #sEMD layer
        semd_lr = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label = "sEMD lr")
        semd_rl = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label = "sEMD rl")
        semd_tb = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label = "sEMD tb")
        semd_bt = p.Population(Sizes.pop_semd, p.extra_models.IF_curr_exp_sEMD, cell_params_semd, label = "sEMD bt")

        if (cleaning_layerFLAG):

            #CLEANING layer
            cleaning_layer_lr = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label = "cleaning lr")
            cleaning_layer_rl = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label = "cleaning rl")
            cleaning_layer_tb = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label = "cleaning tb")
            cleaning_layer_bt = p.Population(Sizes.pop_semd, p.IF_curr_exp, cell_params_semd, label = "cleaning bt")


        # Winner Take All - Global
        if (wtaL):
            wta = p.Population(Sizes.pop_WTA_Local, p.IF_curr_exp, cell_params_wta, label="WTA")


    neurons_row = int(np.sqrt(length_neurons))  # neurons per row full retina
    neighbours = []
    if(wtaL):
        #patch
        hood = range(Sizes.edge_WTALocal)
        for y in range(0, Sizes.edge_WTALocal):
            for x in (0, Sizes.edge_WTALocal):
                hood.append((y*neurons_row)+x)
        for removal in range(0, Sizes.edge_WTALocal):
            hood.remove(removal)
    else:
        hood=[-1, 0, 1]

    if(wtaL):
        # CONNECTION MATRIX, PREPARE COMPLEX CONNECTION MATRIX WTA
        scaling_factorWTA = Sizes.edge_WTALocal
        wta_patch = np.arange(Sizes.pop_WTA_Local)
        wta_patch.resize(int(np.sqrt(Sizes.pop_WTA_Local)), int(np.sqrt(Sizes.pop_WTA_Local)))
        size=(int(scaling_factorWTA), int(scaling_factorWTA))
        wta_patch = np.kron(wta_patch, np.ones(size))
        print wta_patch
        wta_patch.resize(Sizes.pop_semd)


    LRremove = range((neurons_row-1), neurons_row**2, neurons_row)
    RLremove = range(0, neurons_row**2, neurons_row)

    delay_fac=1
    delay_cleaning = 0


    class Connect:
        pass
        #sptc = [(i,sptc[i],Weights.sptc,delay) for i in range(Sizes.pop_dvs/2)]
        # sptc = [(i,i,Weights.sptc,delay) for i in range(Sizes.pop_dvs/2)]

        #LR
        semd_lr_fac = [(i, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd)]
        semd_lr_fac = np.delete(semd_lr_fac, LRremove, 0)

        semd_lr_trig = [(i+1, i, Weights.semd, delay) for i in range(Sizes.pop_semd)]
        semd_lr_trig = np.delete(semd_lr_trig, LRremove, 0)

        #RL
        semd_rl_fac = [(i+1, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd)]
        semd_rl_fac = np.delete(semd_rl_fac, RLremove, 0)

        semd_rl_trig = [(i, i, Weights.semd, delay) for i in range(Sizes.pop_semd)]
        semd_rl_trig = np.delete(semd_rl_trig, RLremove, 0)

        #TB
        semd_tb_fac = [(i, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd)]
        semd_tb_trig = [(i+neurons_row, i, Weights.semd, delay) for i in range(Sizes.pop_semd)]
        #BT
        semd_bt_fac = [(i+neurons_row, i, Weights.semd, delay+delay_fac) for i in range(Sizes.pop_semd)]
        semd_bt_trig = [(i, i, Weights.semd, delay) for i in range(Sizes.pop_semd)]

        if (cleaning_layerFLAG):
            #CLEANING LAYER POPULATION
            cleaning = [(i, i, Weights.cleaning, delay) for i in range(Sizes.pop_semd)]
            #patch connections
            for j in hood:
                semd_clean_inhibit = [(i, i+j, Weights.cleaning, delay + delay_cleaning) for i in range(1, Sizes.pop_semd-1)]
                neighbours.append(semd_clean_inhibit)

        if (wtaL):
            wtaEXT=[(i, wta_patch[i], Weights.WTA_EXT, 0) for i in range(Sizes.pop_semd)]
            wtaINH =[]
            for x in hood:
                y_row = 0
                run = []
                for y in range(Sizes.pop_WTA_Local):
                    if(y > 0 and y % (np.sqrt(Sizes.pop_semd)/Sizes.edge_WTALocal) == 0):
                        y_row+=Sizes.edge_WTALocal
                    run.append([y,x+y*Sizes.edge_WTALocal*y_row*neurons_row, Weights.WTA_Inh, 0])
                npa=np.asarray(run, dtype=np.float)
                wtaINH.append(npa)

    # projections and connections
    # p.Projection(Populations.dvs, Populations.sptc, p.FromListConnector(Connect.sptc), \
    # p.StaticSynapse(), receptor_type= 'excitatory')


    #PROJECTION sEMD
    # HERE sptc, Populations.sptc
    p.Projection(Populations.dvs, Populations.semd_lr, p.FromListConnector(Connect.semd_lr_fac)\
    , p.StaticSynapse(), receptor_type= 'excitatory')
    p.Projection(Populations.dvs, Populations.semd_lr, p.FromListConnector(Connect.semd_lr_trig)\
    , p.StaticSynapse(), receptor_type= 'excitatory2')
    p.Projection(Populations.dvs, Populations.semd_rl, p.FromListConnector(Connect.semd_rl_fac)\
    , p.StaticSynapse(), receptor_type= 'excitatory')
    p.Projection(Populations.dvs, Populations.semd_rl, p.FromListConnector(Connect.semd_rl_trig)\
    , p.StaticSynapse(), receptor_type= 'excitatory2')
    p.Projection(Populations.dvs, Populations.semd_tb, p.FromListConnector(Connect.semd_tb_fac)\
    , p.StaticSynapse(), receptor_type= 'excitatory')
    p.Projection(Populations.dvs, Populations.semd_tb, p.FromListConnector(Connect.semd_tb_trig)\
    , p.StaticSynapse(), receptor_type= 'excitatory2')
    p.Projection(Populations.dvs, Populations.semd_bt, p.FromListConnector(Connect.semd_bt_fac)\
    , p.StaticSynapse(), receptor_type= 'excitatory')
    p.Projection(Populations.dvs, Populations.semd_bt, p.FromListConnector(Connect.semd_bt_trig)\
    , p.StaticSynapse(), receptor_type= 'excitatory2')

    if (cleaning_layerFLAG):

        #PROJECTION CLEANING LAYER, 1st layer
        p.Projection(Populations.semd_lr, Populations.cleaning_layer_lr, p.FromListConnector(Connect.cleaning)\
        , p.StaticSynapse(), receptor_type= 'excitatory')
        p.Projection(Populations.semd_rl, Populations.cleaning_layer_rl, p.FromListConnector(Connect.cleaning)\
        , p.StaticSynapse(), receptor_type= 'excitatory')
        p.Projection(Populations.semd_tb, Populations.cleaning_layer_tb, p.FromListConnector(Connect.cleaning)\
        , p.StaticSynapse(), receptor_type= 'excitatory')
        p.Projection(Populations.semd_bt, Populations.cleaning_layer_bt, p.FromListConnector(Connect.cleaning)\
        , p.StaticSynapse(), receptor_type= 'excitatory')


    # Global inhibition

    if (wtaL):
        p.Projection(Populations.semd_tb, Populations.wta, p.FromListConnector(Connect.wtaEXT), p.StaticSynapse(),
                     receptor_type='excitatory')
        p.Projection(Populations.semd_bt, Populations.wta, p.FromListConnector(Connect.wtaEXT), p.StaticSynapse(),
                     receptor_type='excitatory')
        p.Projection(Populations.semd_rl, Populations.wta, p.FromListConnector(Connect.wtaEXT), p.StaticSynapse(),
                     receptor_type='excitatory')
        p.Projection(Populations.semd_lr, Populations.wta, p.FromListConnector(Connect.wtaEXT), p.StaticSynapse(),
                     receptor_type='excitatory')

        for j in Connect.wtaINH:
            p.Projection(Populations.wta, Populations.semd_tb, p.FromListConnector(j), p.StaticSynapse(),
                         receptor_type='inhibitory')
            p.Projection(Populations.wta, Populations.semd_bt, p.FromListConnector(j), p.StaticSynapse(),
                         receptor_type='inhibitory')
            p.Projection(Populations.wta, Populations.semd_rl, p.FromListConnector(j), p.StaticSynapse(),
                         receptor_type='inhibitory')
            p.Projection(Populations.wta, Populations.semd_lr, p.FromListConnector(j), p.StaticSynapse(),
                         receptor_type='inhibitory')


    # NO PATCH
    # p.Projection(Populations.semd_lr, Populations.cleaning_layer_rl, p.FromListConnector(Connect.cleaning) \
    #              , p.StaticSynapse(), receptor_type='inhibitory')
    # p.Projection(Populations.semd_rl, Populations.cleaning_layer_lr, p.FromListConnector(Connect.cleaning) \
    #              , p.StaticSynapse(), receptor_type='inhibitory')
    # p.Projection(Populations.semd_tb, Populations.cleaning_layer_bt, p.FromListConnector(Connect.cleaning) \
    #              , p.StaticSynapse(), receptor_type='inhibitory')
    # p.Projection(Populations.semd_bt, Populations.cleaning_layer_tb, p.FromListConnector(Connect.cleaning) \
    #              , p.StaticSynapse(), receptor_type='inhibitory')

    if (cleaning_layerFLAG):

        # #PATCH
        for i in neighbours:
            p.Projection(Populations.semd_lr, Populations.cleaning_layer_rl, p.FromListConnector(i)\
            , p.StaticSynapse(), receptor_type= 'inhibitory')
            p.Projection(Populations.semd_rl, Populations.cleaning_layer_lr, p.FromListConnector(i)\
            , p.StaticSynapse(), receptor_type= 'inhibitory')
            p.Projection(Populations.semd_tb, Populations.cleaning_layer_bt, p.FromListConnector(i)\
            , p.StaticSynapse(), receptor_type= 'inhibitory')
            p.Projection(Populations.semd_bt, Populations.cleaning_layer_tb, p.FromListConnector(i)\
            , p.StaticSynapse(), receptor_type= 'inhibitory')



    # RECORDS
    Populations.dvs.record(['spikes']) #HERE sptc
    Populations.semd_lr.record(['spikes'])#,'gsyn_exc','gsyn_inh'])
    Populations.semd_rl.record(['spikes'])#,'gsyn_exc','gsyn_inh'])
    Populations.semd_tb.record(['spikes'])#,'gsyn_exc','gsyn_inh'])
    Populations.semd_bt.record(['spikes'])#,'gsyn_exc','gsyn_inh'])

    if (cleaning_layerFLAG):

        Populations.cleaning_layer_rl.record(['spikes'])#,'gsyn_exc','gsyn_inh'])
        Populations.cleaning_layer_lr.record(['spikes'])#,'gsyn_exc','gsyn_inh'])
        Populations.cleaning_layer_tb.record(['spikes'])#,'gsyn_exc','gsyn_inh'])
        Populations.cleaning_layer_bt.record(['spikes'])#,'gsyn_exc','gsyn_inh'])
    if(wtaL):
        Populations.wta.record(['v'])

    # run simulation
    p.run(simulation_runtime)

    class Data:
        pass
    # RECEIVING DATA FROM SIMULATION
        sptc_spikes = Populations.dvs.get_data(['spikes']) #HERE sptc
        semd_lr_spikes = Populations.semd_lr.get_data(['spikes'])
        semd_rl_spikes = Populations.semd_rl.get_data(['spikes'])
        semd_bt_spikes = Populations.semd_bt.get_data(['spikes'])
        semd_tb_spikes = Populations.semd_tb.get_data(['spikes'])

        if (cleaning_layerFLAG):

            cleaning_layer_rl_spikes = Populations.cleaning_layer_rl.get_data(['spikes'])
            cleaning_layer_lr_spikes = Populations.cleaning_layer_lr.get_data(['spikes'])
            cleaning_layer_bt_spikes = Populations.cleaning_layer_bt.get_data(['spikes'])
            cleaning_layer_tb_spikes = Populations.cleaning_layer_tb.get_data(['spikes'])
        if(wtaL):
            wtaV = Populations.wta.get_data(['v'])


    if (multiple_speedsFLAG):
        print 'multiple speed processing for speed ' + str(bar_speed)
        # spks_lr = 0
        # spks_rl = 0
        # spks_tb = 0
        # spks_bt = 0
        spks_lr = []
        spks_rl = []
        spks_tb = []
        spks_bt = []

        # RASTER PLOT sEMD
        for i in range(Sizes.pop_semd):
            # ax5.plot(Data.sptc_spikes.segments[0].spiketrains[i], [i] * len(Data.sptc_spikes.segments[0].spiketrains[i]),".b")
            # spks_sptc = spks_sptc	+len(Data.sptc_spikes.segments[0].spiketrains[i])
            # spks_lr = spks_lr + len(Data.semd_lr_spikes.segments[0].spiketrains[i])
            # spks_rl = spks_rl + len(Data.semd_rl_spikes.segments[0].spiketrains[i])
            # spks_tb = spks_tb + len(Data.semd_tb_spikes.segments[0].spiketrains[i])
            # spks_bt = spks_bt + len(Data.semd_bt_spikes.segments[0].spiketrains[i])
            spks_lr.append(float(len(Data.semd_lr_spikes.segments[0].spiketrains[i])) / total_period * 1000)
            spks_rl.append(float(len(Data.semd_rl_spikes.segments[0].spiketrains[i])) / total_period * 1000)
            spks_tb.append(float(len(Data.semd_tb_spikes.segments[0].spiketrains[i])) / total_period * 1000)
            spks_bt.append(float(len(Data.semd_bt_spikes.segments[0].spiketrains[i])) / total_period * 1000)

        # spks_lr = spks_lr / float(total_period) * 1000.0
        # spks_rl = spks_rl / float(total_period) * 1000.0
        # spks_tb = spks_tb / float(total_period) * 1000.0
        # spks_bt = spks_bt / float(total_period) * 1000.0

        #data from all the population
        LR.append(spks_lr)
        RL.append(spks_rl)
        TB.append(spks_tb)
        BT.append(spks_bt)

    else:

        print 'saving all data ...'

        [data_spks_lr, data_spks_rl, data_spks_tb, data_spks_bt, data_spks_cleaning_lr, data_spks_cleaning_rl, data_spks_cleaning_tb, data_spks_cleaning_bt]=saving_data_sEMDandCLEANING(saving_data_path, Data, Sizes, total_period, direction, cleaning_layerFLAG)
        [spks_lr, spks_rl, spks_tb, spks_bt, spks_cleaning_lr, spks_cleaning_rl, spks_cleaning_tb, spks_cleaning_bt]=plotting_sEMDandCLEANING(plot_FLAG, Data, Sizes, simulation_plot, total_period, bar_speed, direction, stimulus, cleaning_layerFLAG, length_neurons)

        #data from all the population
        LR.append(spks_lr)
        RL.append(spks_rl)
        TB.append(spks_tb)
        BT.append(spks_bt)

        if (cleaning_layerFLAG):

            CLEANING_LR.append(spks_cleaning_lr)
            CLEANING_RL.append(spks_cleaning_rl)
            CLEANING_TB.append(spks_cleaning_tb)
            CLEANING_BT.append(spks_cleaning_bt)

        #detailed data
        LR_data.append(data_spks_lr)
        RL_data.append(data_spks_rl)
        TB_data.append(data_spks_tb)
        BT_data.append(data_spks_bt)

        if (cleaning_layerFLAG):

            CLEANING_LR_data.append(data_spks_cleaning_lr)
            CLEANING_RL_data.append(data_spks_cleaning_rl)
            CLEANING_TB_data.append(data_spks_cleaning_tb)
            CLEANING_BT_data.append(data_spks_cleaning_bt)

    p.end()


    if multiple_speedsFLAG:
        name_file = 'LR'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(LR, f)

        print 'LR saved'

        name_file = 'RL'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(RL, f)

        print 'RL saved'

        name_file = 'TB'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(TB, f)

        print 'TB saved'

        name_file = 'BT'+str(bar_speed)
        path_file = saving_data_path + name_file + ".pickle"
        with open(path_file, "wb") as f:
            pickle.dump(BT, f)

        print 'BT saved'

if not multiple_speedsFLAG:
    name_file = 'LR'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(LR, f)

    print 'LR saved'

    name_file = 'RL'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(RL, f)

    print 'RL saved'

    name_file = 'TB'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(TB, f)

    print 'TB saved'

    name_file = 'BT'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(BT, f)

    print 'BT saved'

if (cleaning_layerFLAG):

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

    print 'cleaning saved'

#detailed data
name_file = 'LRdata'
path_file = saving_data_path + name_file + ".pickle"
with open(path_file, "wb") as f:
    pickle.dump(LR_data, f)

name_file = 'RLdata'
path_file = saving_data_path + name_file + ".pickle"
with open(path_file, "wb") as f:
    pickle.dump(RL_data, f)

name_file = 'TBdata'
path_file = saving_data_path + name_file + ".pickle"
with open(path_file, "wb") as f:
    pickle.dump(TB_data, f)

name_file = 'BTdata'
path_file = saving_data_path + name_file + ".pickle"
with open(path_file, "wb") as f:
    pickle.dump(BT_data, f)

print 'detailed data saved'

if (cleaning_layerFLAG):

    name_file='CLEANING_LRdata'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(CLEANING_LR_data, f)

    name_file='CLEANING_RLdata'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(CLEANING_RL_data, f)

    name_file='CLEANING_TBdata'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(CLEANING_TB_data, f)

    name_file='CLEANING_BTdata'
    path_file = saving_data_path + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(CLEANING_BT_data, f)

    print 'cleaning layer detailed data saved'

name_file='BAR_SPEED'
path_file = saving_data_path + name_file + ".pickle"
with open(path_file, "wb") as f:
    pickle.dump(BAR_SPEED, f)


# fig1 = plt.figure()
# plt.plot(BAR_SPEED, CLEANING_LR, 'r--', BAR_SPEED, CLEANING_RL, 'b--', BAR_SPEED, CLEANING_TB, 'g--',  CLEANING_BT, 'm--')
# plt.xlabel('bar speed [px/ms]')
# plt.ylabel('Firing Rate [Hz]')
# fig1.savefig('Cleaning Layer Neurons Firing Rate vs Bar Speed')
if(wtaL):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(range(len(Data.wtaV.segments[0].filter(name = 'v')[0])), Data.wtaV.segments[0].filter(name = 'v')[0])
if (plot_FLAG):
    plt.show()




