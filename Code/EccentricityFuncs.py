#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:44:53 2019

@author: giulia.dangelo@iit.it
"""

import math as mt
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import pickle
from tqdm import tqdm
import time
from random import seed
from random import randint

class RF:

    def __init__(self):
        self.mem_pot=float(0)
        self.num_px_threshold=float(0)
        self.threshold=float(0)
        self.previous_timestamp=float(0)
        self.half_size=0
        self.size=0
        self.center_x=0
        self.center_y=0
        self.mask_TL_x=0
        self.mask_TL_y=0
        self.diag=0
        self.mask_starting_point=0
        self.ring_mask=0
        self.id_trajectory=0
        self.RF_id=0
        self.RF_neuron_id=[]
        self.RF_neuron_id_x=[]
        self.RF_neuron_id_y=[]
        self.spikes=[]
        self.num_active_px=0


def create_mask(RFs, maximum_kernel_size, width, height, starting_point, ring_mask, RFthreshold_percentage, tau):

    step=int(maximum_kernel_size)-1
    old_sizeRFs=len(RFs)
    
    starting_point=int(starting_point)
    height=int(height)
    width=int(width)
    step=int(step)
    
    
    #LEFT: from top to bottom
    x=starting_point
    for y in range(starting_point,(height-starting_point+int(mt.ceil(maximum_kernel_size/2))),step):
        RFs_tmp=RF()
        RFs_tmp.center_y=y
        RFs_tmp.center_x=x
        RFs.append(RFs_tmp)
    

    #BOTTOM: from left to right
    y=(height-starting_point)
    for x in range(starting_point,(width-starting_point+int(mt.ceil(maximum_kernel_size/2))),step):
        RFs_tmp=RF()
        RFs_tmp.center_y=y
        RFs_tmp.center_x=x
        RFs.append(RFs_tmp)
     

    #TOP: from left to right
    y=starting_point
    for x in range(starting_point,(width-starting_point+int(mt.ceil(maximum_kernel_size/2))),step):
        RFs_tmp=RF()
        RFs_tmp.center_y=y
        RFs_tmp.center_x=x
        RFs.append(RFs_tmp)
    

    #RIGHT: from top to bottom    
    x=(width-starting_point)
    for y in range(starting_point,(height-starting_point+int(mt.ceil(maximum_kernel_size/2))),step):
        RFs_tmp=RF();
        RFs_tmp.center_y=y
        RFs_tmp.center_x=x
        RFs.append(RFs_tmp)


    for i in range(old_sizeRFs,len(RFs)):
        RFs[i].size=maximum_kernel_size
        RFs[i].half_size=float(maximum_kernel_size)/2
        RFs[i].diag=mt.ceil(mt.sqrt(mt.pow(RFs[i].size,2)+mt.pow(RFs[i].size,2)))
        RFs[i].mask_starting_point=starting_point
        RFs[i].mask_TL_x=int(np.floor(starting_point-RFs[i].half_size))
        RFs[i].mask_TL_y=int(np.floor(starting_point-RFs[i].half_size))
        RFs[i].threshold=float(1)
        RFs[i].num_px_threshold=(RFs[i].size*RFs[i].size)*RFthreshold_percentage/100
        RFs[i].ring_mask=ring_mask
        RFs[i].RF_id=i

    return RFs


def RFsEccentricity(maximum_kernel_size, camera_resolution, fovea_dimensions, RFthreshold_percentage, tau):
    next_kernel_sizeTOT=[]
    mask_ringsTL=[]
    rings_starting_point=[]
    #PERIPHERY
    foveaTLx=int(mt.ceil((camera_resolution[0]/2)-(fovea_dimensions[0]/2)))
    foveaTLy=int(mt.ceil((camera_resolution[1]/2)-(fovea_dimensions[1]/2)))

    
    #FIRST MASK
    starting_point=int(mt.ceil(maximum_kernel_size/2))
    first_mask=1
    RFs=[]
    RFs = create_mask(RFs, maximum_kernel_size, camera_resolution[0], camera_resolution[1], starting_point, first_mask, RFthreshold_percentage, tau)
    max_eccentricity_x=foveaTLx-RFs[-1].mask_TL_x
    
    RFs_counter=len(RFs)
    ring_counter=2

    while(RFs[RFs_counter-1].half_size>=2):
        #next mask
        eccentricity_x=RFs[RFs_counter-1].mask_starting_point+RFs[RFs_counter-1].half_size #assuming square camera resolution I can only the x axis
        #eccentricity_y=foveaTLx-RFs[RFs_counter-1].mask_starting_point;
        #Linear RF size decrasing
        next_kernel_size=int(-(float(maximum_kernel_size)/max_eccentricity_x)*eccentricity_x)+int(maximum_kernel_size)
        next_kernel_sizeTOT.append(next_kernel_size)
        rings_starting_point.append(RFs[RFs_counter-1].mask_starting_point)
        next_starting_point=np.floor(RFs[RFs_counter-1].mask_starting_point+RFs[RFs_counter-1].half_size)
        RFs = create_mask(RFs, next_kernel_size, camera_resolution[0], camera_resolution[1], next_starting_point, ring_counter, RFthreshold_percentage, tau)
        RFs_counter=len(RFs)
        ring_counter=ring_counter+1
        mask_ringsTL.append(RFs[RFs_counter-1].mask_TL_x)

    #FOVEA
    starting_point_fovea_x=int(RFs[-1].mask_TL_x+RFs[-1].half_size)
    starting_point_fovea_y=int(RFs[-1].mask_TL_y+RFs[-1].half_size)

    old_sizeRFs=len(RFs)
    for x in range(starting_point_fovea_x,(camera_resolution[0]-starting_point_fovea_x)):
        for y in range(starting_point_fovea_y,(camera_resolution[1]-starting_point_fovea_y)):
            RFs_fovea_tmp=RF()
            
            RFs_fovea_tmp.center_x=x
            RFs_fovea_tmp.center_y=y

            RFs_fovea_tmp.size=1
            RFs_fovea_tmp.half_size=0.5
            RFs_fovea_tmp.diag=1
            RFs_fovea_tmp.mask_starting_point=[starting_point_fovea_x, starting_point_fovea_y]
            RFs_fovea_tmp.mask_TL_x=int(starting_point_fovea_x)
            RFs_fovea_tmp.mask_TL_y=int(starting_point_fovea_y)
            RFs_fovea_tmp.ring_mask=ring_counter
            RFs_fovea_tmp.RF_id=old_sizeRFs
            RFs_fovea_tmp.threshold = float(1)
            RFs_fovea_tmp.num_px_threshold=1
            old_sizeRFs=old_sizeRFs+1
            RFs.append(RFs_fovea_tmp)
    print ("Eccentricity Layer created!")
    return RFs



def fovea_percentage(fovea_percentage, camera_resolution):
    fovea_width=mt.ceil(fovea_percentage*camera_resolution[0]/100)
    fovea_height=mt.ceil(fovea_percentage*camera_resolution[1]/100)
    fovea_dimensions=[fovea_width, fovea_height]
    
    return fovea_dimensions

def display_eccentricity_rings_coordinates(RFs, camera_resolution, folder_checking_eccentricity,saving_data_path):
    vSurface = np.zeros(camera_resolution)
    colors = mt.floor(255/(RFs[-1].ring_mask-1))
    # displaying the eccentricity with different colours for each ring
    for i in range(0, len(RFs)):
        x_start = int(mt.ceil(abs(RFs[i].center_y-RFs[i].half_size)+1))
        x_end = int(mt.ceil(abs(RFs[i].center_y+RFs[i].half_size)))
        
        y_start = int(mt.ceil(abs(RFs[i].center_x-RFs[i].half_size)+1))
        y_end = int(mt.ceil(abs(RFs[i].center_x+RFs[i].half_size)))
        
        if x_start == x_end and y_start == y_end:
            x = x_end
            y = y_end
            vSurface[x, y] = RFs[i].ring_mask*colors
        vSurface[x_start:x_end, y_start:y_end] = RFs[i].ring_mask*colors

    # creating a list with all the TopLeft points for each ring
    mask_TL_x = []
    mask_TL_y = []

    old_x = int(RFs[0].mask_TL_x)
    old_y = int(RFs[0].mask_TL_y)
    mask_TL_x.append(old_x)
    mask_TL_y.append(old_y)
    for i in range(1, len(RFs)):
        curr_x = int(RFs[i].mask_TL_x)
        curr_y = int(RFs[i].mask_TL_y)

        if curr_x != old_x and curr_y != old_y:
            mask_TL_x.append(curr_x)
            mask_TL_y.append(curr_y)
            old_x = curr_x
            old_y = curr_y

        vSurface[curr_x, curr_y]=255
    scipy.misc.imsave(folder_checking_eccentricity+'rings_plus_TLpoints.jpg', vSurface)
    
    # finding the x,y coordinates to draw the horizontal trajectories in between two TL ring points
    coordinates_x = []
    coordinates_y = []
    
    pre_coord_x = mask_TL_x[0]
    for i in range(1, len(mask_TL_x)):
        coordinates_x.append(mask_TL_x[i-1]+int(mt.floor((mask_TL_x[i]-pre_coord_x)/2)))
        pre_coord_x=mask_TL_x[i]
    coordinates_x.append(pre_coord_x) #fovea
    
    pre_coord_y=mask_TL_y[0]
    for i in range(1, len(mask_TL_y)):
        coordinates_y.append(mask_TL_y[i-1]+int(mt.floor((mask_TL_y[i]-pre_coord_y)/2)))
        pre_coord_y=mask_TL_y[i]
    coordinates_y.append(pre_coord_y) #fovea

    for i in range(0, len(coordinates_y)):
         vSurface[coordinates_y[i], 0:camera_resolution[0]]=255
         vSurface[camera_resolution[1]-coordinates_y[i], 0:camera_resolution[0]]=255
    vSurface[coordinates_y[-1]:camera_resolution[1]-coordinates_y[-1], 0:camera_resolution[0]]=255
    scipy.misc.imsave(folder_checking_eccentricity+'trjectories_lines.jpg', vSurface)

    # creating an array containing all the TL rings points
    y_trajectories = []
    for x in range(0, len(mask_TL_x)): # start to fovea
        y_trajectories.append(mask_TL_x[x])
    for x in range((mask_TL_x[-1]+1), (camera_resolution[0]-mask_TL_x[-1])): # fovea
        y_trajectories.append(x)
    for y in range(0, len(mask_TL_y)): # fovea to the end
        y_trajectories.append(camera_resolution[0]-mask_TL_x[(len(mask_TL_x)-1)-y])

    trajectories_path_file = saving_data_path + 'trajectories.pickle'
    with open(trajectories_path_file, "wb") as f:
        pickle.dump(y_trajectories, f)

    print ("Eccentricity displayed !")
    return coordinates_x, coordinates_y, y_trajectories


def has_connection_xy_trajectories(RFs, y_trajectories, camera_resolution, saving_data_path):
    print ("has connection started ...")
    seed(1)

    counter_y=0
    for y_traj in range(0, (len(y_trajectories)-1)):
        vSurface = np.zeros(camera_resolution)
        y_start = y_trajectories[y_traj]
        y_end = y_trajectories[y_traj+1]
        for y in range(y_start, y_end):
            for x in range(0, camera_resolution[0]):
                for rf in RFs:
                    if abs(rf.center_x-x) <= rf.half_size and abs(rf.center_y-y) <= rf.half_size:
                        rf.id_trajectory = y_trajectories[y_traj]
                        if not counter_y in rf.RF_neuron_id_y:
                            rf.RF_neuron_id_y.append(counter_y)
                        vSurface[int(rf.center_y-rf.half_size):int(rf.center_y+rf.half_size), int(rf.center_x-rf.half_size):int(rf.center_x+rf.half_size)] = randint(0, 255)
        counter_y += 1
        print (str(y_traj*100/(len(y_trajectories)-1)) + '% ...loading y trajectories ')
        scipy.misc.imsave('/localhome/mbaxrap7/giulia/tde-iCub/Code/sEMD/CheckingEccentricity/horizontal_trajectories/' + 'trajectory_hor' + str(counter_y) + '.jpg', vSurface)

    counter_x=0
    for y_traj in range(0, (len(y_trajectories)-1)):
        vSurface = np.zeros(camera_resolution)
        x_start = y_trajectories[y_traj]
        x_end = y_trajectories[y_traj+1]
        for x in range(x_start, x_end):
            for y in range(0, camera_resolution[0]):
                for rf in RFs:
                    if abs(rf.center_x-x) <= rf.half_size and abs(rf.center_y-y) <= rf.half_size:
                        rf.id_trajectory = y_trajectories[y_traj]
                        if not counter_x in rf.RF_neuron_id_x:
                            rf.RF_neuron_id_x.append(counter_x)
                        vSurface[int(rf.center_y-rf.half_size):int(rf.center_y+rf.half_size), int(rf.center_x-rf.half_size):int(rf.center_x+rf.half_size)] = randint(0, 255)
        counter_x += 1
        print (str(y_traj*100/(len(y_trajectories)-1)) + '% ...loading x trajectories ')
        scipy.misc.imsave('/localhome/mbaxrap7/giulia/tde-iCub/Code/sEMD/CheckingEccentricity/vertical_trajectories/' + 'trajectory_ver' + str(counter_x) + '.jpg', vSurface)

    print ('counter_x' + str(counter_x) + 'counter_y' + str(counter_y))

    counter_x_path_file = saving_data_path + 'counter_x.pickle'
    with open(counter_x_path_file, "wb") as f:
        pickle.dump(counter_x, f)

    counter_y_path_file = saving_data_path + 'counter_y.pickle'
    with open(counter_y_path_file, "wb") as f:
        pickle.dump(counter_y, f)

    return RFs, counter_x, counter_y

def has_connection_trajectories(RFs, camera_resolution, counter_x, counter_y, visualise_neuronsFLAG):
    seed(1)
    counter_neuronID=0
    for y in range(0, counter_y):
        for x in range(0, counter_x):
            for rf in RFs:
                if x in rf.RF_neuron_id_x and y in rf.RF_neuron_id_y:
                    if not counter_neuronID in rf.RF_neuron_id:
                        rf.RF_neuron_id.append(counter_neuronID)
            counter_neuronID+=1


    print ("trajectories connection found !")
    length_neurons=counter_x*counter_y

    if(visualise_neuronsFLAG):
        counter_fig =0
        for id in range (0,length_neurons):
            vSurface = np.zeros(camera_resolution)
            for rf in RFs:
                if id in rf.RF_neuron_id:
                    vSurface[int(rf.center_y - rf.half_size):int(rf.center_y + rf.half_size),int(rf.center_x - rf.half_size):int(rf.center_x + rf.half_size)] = randint(0, 255)
            scipy.misc.imsave('/localhome/mbaxrap7/giulia/tde-iCub/Code/sEMD/CheckingEccentricity/neuronID/' + 'neuronID' + str(counter_fig) + '.jpg', vSurface)
            counter_fig += 1

    return RFs, length_neurons

            
def trajectories_neuron_id(RFs, y_trajectories, camera_resolution, folder_checking_eccentricity):
    name_img='trajectories'
    extension='.jpg'
    vSurface=np.zeros(camera_resolution)
    seed(1)
    for traj_index in range(0, len(y_trajectories)):
        for RF_index in range(0, len(RFs)):
            if(RFs[RF_index].id_trajectory==y_trajectories[traj_index]):
                x_start=int(mt.ceil(abs(RFs[RF_index].center_y-RFs[RF_index].half_size)+1))
                x_end=int(mt.ceil(abs(RFs[RF_index].center_y+RFs[RF_index].half_size)))
                
                y_start=int(mt.ceil(abs(RFs[RF_index].center_x-RFs[RF_index].half_size)+1))
                y_end=int(mt.ceil(abs(RFs[RF_index].center_x+RFs[RF_index].half_size)))
                
                if (x_start==x_end and y_start==y_end):
                    x=x_end
                    y=y_end
                    vSurface[x, y] = randint(0, 255)
                vSurface[x_start:x_end,y_start:y_end]= randint(0, 255)
    scipy.misc.imsave(folder_checking_eccentricity+name_img+extension, vSurface)

def checking_trajectories(RFs, y_trajectories, camera_resolution, folder_checking_eccentricity):
    name_img='trject'
    extension='.jpg'
    RFs_trajectories=[]
    for traj_index in range(0,len(y_trajectories)):
        vSurface=np.zeros(camera_resolution)
        RFs_trajectory_traj_index=[]
        for RF_index in range(0,len(RFs)):
            if(RFs[RF_index].id_trajectory==y_trajectories[traj_index]):
                RFs_trajectory_traj_index.append(RFs[RF_index].RF_id)

                x_start=int(mt.ceil(abs(RFs[RF_index].center_y-RFs[RF_index].half_size)+1))
                x_end=int(mt.ceil(abs(RFs[RF_index].center_y+RFs[RF_index].half_size)))
                
                y_start=int(mt.ceil(abs(RFs[RF_index].center_x-RFs[RF_index].half_size)+1))
                y_end=int(mt.ceil(abs(RFs[RF_index].center_x+RFs[RF_index].half_size)))
                
                if (x_start==x_end and y_start==y_end):
                    x=x_end
                    y=y_end
                    vSurface[x,y]=255
                else:
                    vSurface[x_start:x_end,y_start:y_end]=255
        RFs_trajectories.append(RFs_trajectory_traj_index)
        scipy.misc.imsave(folder_checking_eccentricity+name_img+str(traj_index)+extension, vSurface)
    print("trajectories created!")
    return RFs_trajectories              
            
            
def checking_numberRFs(RFs_trajectories):
    countingRFs=0
    for index in range(0, len(RFs_trajectories)):
        countingRFs_tmp=len(RFs_trajectories[index])
        countingRFs=countingRFs+countingRFs_tmp
    return countingRFs   
            
def reorderingRFs_neuron_id(RFs, RFs_trajectories):            
    RF_counter=0
    NEW_RFs=[]
    for index in range(0, len(RFs_trajectories)):
        RFs_trajectories_tmp=RFs_trajectories[index]
        for index_RFtmp in range (0,len(RFs_trajectories_tmp)):
            for RF_index in range(0,len(RFs)):
                if(RFs[RF_index].RF_id==RFs_trajectories_tmp[index_RFtmp]):
                    RFs[RF_index].RF_neuron_id=RF_counter
                    NEW_RFs.append(RFs[RF_index])
                    RF_counter=RF_counter+1
    print( "neurons reordered!")
    return NEW_RFs
            
            
def RFcell(events, RFs, path_file, tau, length_neurons):
    spikes=[]
    neurons=[[] for _ in range(length_neurons)]
    name_file="neurons"

    for e in tqdm(events):
        for rf in RFs:
            if (abs(rf.center_x-e.x) <= rf.half_size) and (abs(rf.center_y-e.y) <= rf.half_size):
                #dt from the previous timestamp within the RF
                if (rf.size ==1):
                    dt=0
                else:
                    dt=float(e.timestamp)-float(rf.previous_timestamp)
                    rf.previous_timestamp=e.timestamp

                #counting active pixel within the RF
                rf.mem_pot = float(rf.mem_pot) * np.exp(-dt / float(tau))
                rf.mem_pot += float(1.0/(rf.num_px_threshold))

                if(rf.mem_pot>=rf.threshold):
                    rf.mem_pot=float(0)
                    rf.spikes.append(e.timestamp)
                    spikes.append(rf.RF_id)
                    for items in rf.RF_neuron_id:
                        neurons[items].append(e.timestamp)
    # sorting the timestamps
    for i in range(0, len(neurons)):
        neuron_tmp=np.asarray(neurons[i])
        neuron_tmp=quickSort(neuron_tmp, 0, len(neuron_tmp)-1)
        neurons[i]=neuron_tmp

    path_file=path_file + name_file + ".pickle"
    with open(path_file, "wb") as f:
        pickle.dump(neurons, f)
    return RFs, spikes, neurons, path_file
            
def write_neurons_file(neurons, name_file, folder_results):
    print ("writing neurons in file")
    name_file=folder_results+name_file + '.txt'
    with open(name_file, "w") as f:
        for s in neurons:
            f.write(str(s) + "\n")

def load_file(file):
    time.sleep(1)
    with open(file, "rb") as f:
        a = pickle.load(f)
    print ("loaded file correctly!")
    return a
# This function takes last element as pivot, places
# the pivot element at its correct position in sorted
# array, and places all smaller (smaller than pivot)
# to left of pivot and all greater elements to right
# of pivot
def partition(arr, low, high):
    i = (low - 1)  # index of smaller element
    pivot = arr[high]  # pivot

    for j in range(low, high):

        # If current element is smaller than or
        # equal to pivot
        if arr[j] <= pivot:
            # increment index of smaller element
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)


# The main function that implements QuickSort
# arr[] --> Array to be sorted,
# low  --> Starting index,
# high  --> Ending index

# Function to do Quick sort
def quickSort(arr, low, high):
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high)

        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)
    return  arr
            
            