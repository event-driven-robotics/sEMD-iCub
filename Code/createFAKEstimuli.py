import scipy
import numpy as np
from EccentricitysEMDFunctions import *
import os


def check_stimuli(stimuli, folder_path, contrast,camera_resolution, bar_dimensions, bar_speed, events_FLAG):
    folder_name = 'FAKEstimuli/'
    try:
        os.mkdir(folder_path + folder_name)
    except OSError:
        print("Creation of the directory %s failed" % folder_path)
    else:
        print("Successfully created the directory %s " % folder_path)

    for stimulus in stimuli:

        try:
            os.mkdir(folder_path + folder_name + stimulus + '/')
        except OSError:
            print("Creation of the directory %s failed" % folder_path)
        else:
            print("Successfully created the directory %s " % folder_path)

        space_metric = 1  # pixel
        direction = stimulus

        #create events
        [eventsATIS_ON, eventsATIS_OFF, total_period]=FakeStimuliBarMoving(camera_resolution, bar_speed, space_metric, bar_dimensions, direction, contrast)


        #visualise events
        temporal_window_start=15 #ms

        if(events_FLAG=='ON'):
            events=eventsATIS_ON
        elif(events_FLAG=='OFF'):
            events=eventsATIS_OFF

        size=camera_resolution
        vSurface=np.zeros(size)*255

        counter=0
        temporal_window = temporal_window_start
        for event in events:
            t = event.timestamp
            x = event.x
            y = event.y

            vSurface[y-1][x-1]=255

            if t >= temporal_window:
                temporal_window=t+temporal_window_start
                scipy.misc.imsave(folder_path+ folder_name + stimulus+'/'+str(counter) + '.jpg', vSurface)
                counter+=1
                vSurface = np.zeros(size)

    return  eventsATIS_ON, eventsATIS_OFF


# PARAMETERS
camera_resolution = [160, 160]
plot_FLAG = True

camera_resolution = [160, 160]
bar_dimensions = [20, 50]
bar_speed = 0.3 # px/ms

# contrast='BlackOverWhite'
contrast = 'WhiteOverBlack'

# path to save data
folder_path = '/home/giulia/Desktop/'


stimuli = ['LR', 'RL', 'BT', 'TB']
# stimuli = ['TransTlBr', 'TransTrBl', 'TransBlTr', 'TransBrTl']

# events to be shown
events_FLAG = 'ON'
# events_FLAG = 'OFF'

[eventsATIS_ON, eventsATIS_OFF] = check_stimuli(stimuli, folder_path, contrast,camera_resolution, bar_dimensions, bar_speed,events_FLAG)


print ('end')