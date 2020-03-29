from EccentricitysEMDFunctions import *


def check_stimuli(stimuli):
    for stimulus in stimuli:
        direction = stimulus
        BAR_SPEED = [0.3]
        space_metric = 1  # pixel
        camera_resolution = [160, 160]
        bar_dimensions = [160, 160]
        bar_speed=BAR_SPEED[0]

        #contrast='BlackOverWhite'
        contrast='WhiteOverBlack'

        #create events
        [eventsATIS_ON, eventsATIS_OFF, total_period]=FakeStimuliBarMoving(camera_resolution, bar_speed, space_metric, bar_dimensions, direction, contrast)


        #visualise events
        temporal_window_start=15 #ms
        events=eventsATIS_ON

        size=camera_resolution
        vSurface=np.zeros(size)

        counter=0
        temporal_window = temporal_window_start
        for event in events:
            t = event.timestamp
            x = event.x
            y = event.y

            vSurface[y-1][x-1]=255

            if t >= temporal_window:
                temporal_window=t+temporal_window_start
                scipy.misc.imsave(
                    '/home/giulia/Desktop/checkFAKEstimuli/'+stimulus+'/'+str(
                        counter) + '.jpg', vSurface)
                counter+=1
                vSurface = np.zeros(size)


# PARAMETERS
camera_resolution = [160, 160]
plot_FLAG = True


stimuli = ['LR', 'RL', 'BT', 'TB']
# stimuli = ['TransTlBr', 'TransTrBl', 'TransBlTr', 'TransBrTl']
check_stimuli(stimuli)


print 'end'