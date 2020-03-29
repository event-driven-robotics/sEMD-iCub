#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:54:08 2019

@author: giulia
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal
import sys
import os
import shutil


def ATISconvert_data(ATIS_width, ATIS_height, path):
    
    from decode_events import*
    
    #Assuming channel 0 as left camera and channel 1 as right camera
    # Import data.log and Decode events
    dm = DataManager()
    dm.load_AE_from_yarp(path)
    print 'ATIS data processing ended'
    
def ATISload_data(camera_resolution, path):
    # Loading decoded events; data(timestamp, channel, x, y, polarity)  
    stereo_data=np.loadtxt(path, delimiter=' ' )
    [left_data,right_data]=split_stereo_data(stereo_data)
    
    return left_data,right_data



def split_stereo_data(stereo_data):
    width_stereo_data, height_stereo_data=stereo_data.shape
    print 'Loading ATIS data ended'
    
    left_data=[]
    right_data=[]
    
    for i in range(1,width_stereo_data):    
        if stereo_data[i,1]==0:
            left_data.append(stereo_data[i,:])       
        else:
            right_data.append(stereo_data[i,:]) 
    return left_data, right_data

def converting_loadingATISdata(ATIS_width, ATIS_height, dataset_path):
    #ATISconvert_data: convert data.log coming from the zynq to a txt file (data.log.txt)
    ATISconvert_data(ATIS_width, ATIS_height, dataset_path)
    
    #Loading ATIS dataset: event= (timestamp, channel, y, x, polarity) for the left/right channel (left/right camera)
    [left_data,right_data]=ATISload_data(ATIS_width, ATIS_height, dataset_path)
    
    return left_data,right_data

def create_video(file_location, file_name, frame_rate, spike_data ,x_res, y_res, direction):
    frame_duration = 1000. / frame_rate

    print "parsing data"
    list_data = []
    x = []
    y = []
    t = []
    max_time = 0.
    min_time = 1000000.
    for spike in spike_data:
        x.append(spike.x)
        y.append(spike.y)
        t.append(spike.timestamp)
        if t[-1] > max_time:
            max_time = spike.timestamp
        if t[-1] < min_time:
            min_time = spike.timestamp
        list_data.append([x[-1], y[-1], t[-1]])
    # list_data.sort(key=lambda x: x[2])

    print "binning frames"
    binned_frames = [[[0 for y in range(y_res)] for x in range(x_res)] for i in range(int(np.ceil((max_time - min_time) / frame_duration)))]

    # setup toolbar
    toolbar_width = 40
    print '[{}]'.format('-'*toolbar_width)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    current_point = 0
    progression_count = 0
    for spike in list_data:
        x = int(spike[0])
        y = int(spike[1])
        t = spike[2] - min_time
        time_index = int(t / frame_duration)
        binned_frames[time_index][y][x] += 1
        progression_count += 1
        # print progression_count, '/', len(list_data)
        if current_point < int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width))):
            current_point = int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width)))
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("]\n")

    print 'creating images'
    filenames = []
    # setup toolbar
    toolbar_width = 40
    print '[{}]'.format('-'*toolbar_width)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    current_point = 0
    progression_count = 0
    isFile = os.path.isdir(file_location+'/frames')
    if (isFile == 1):
        shutil.rmtree(file_location+'/frames')
    os.mkdir(file_location+'/frames')
    for frame in binned_frames:
        plt.imshow(frame, cmap='hot', interpolation='nearest')
        title = '{} - {}'.format(file_name, binned_frames.index(frame))
        title += '.jpg'
        filenames.append(file_location+'/frames/'+title)
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.savefig(file_location+'/frames/'+title, format='jpeg', bbox_inches='tight')
        # plt.show()
        plt.clf()
        progression_count += 1
        if current_point < int(round((float(progression_count) / float(len(binned_frames))) * float(toolbar_width))):
            current_point = int(round((float(progression_count) / float(len(binned_frames))) * float(toolbar_width)))
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("]\n")

    print "creating video"
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(file_location+'/'+file_name+direction+'.gif', images)

    # with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

