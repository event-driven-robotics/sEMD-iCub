"""
Converter from aedat to spikes
See https://www.cit-ec.de/en/nbs/spiking-insect-vision for more details
"""



# imports
import numpy as np
import struct
import sys
from EccentricitysEMDFunctions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyNN.utility.plotting import Figure, Panel




# function to convert aedat into the two arrays record_on and record_off
# for on and off events.

def aedat_to_spikes(input_file, N_AER_ADRESSES, debug=1):

  def createSpikeSourceArray(aedat):
    list = [[] for i in range(N_AER_ADRESSES)]
    for i in range(0, len(aedat)):
      list[int(aedat[i][0])].append(aedat[i][1])
    return list
  
  
  def loadAedat32(path):
    f = open(path, "rb")
    pos = 0
    evt = 0
    timestamp = 0
    rowID = 0
    position_pointer = 0
    aedatFile = []
    global minTimestamp
    global maxTimestamp
    global aedatLength
    global endTime


    maxTimestamp = 0
    minTimestamp = sys.maxsize
    aedatLength = 0
    endTime = 0
    
    z = 1;
    
    # header
    header_line = f.readline()
    while header_line and header_line[0] == "#":
        position_pointer += len(header_line)
        header_line = f.readline()
        if debug >= 2:
            print str(header_line)
        continue
        
    try:
      while True:
  
        f.seek(position_pointer) 
        lec = f.read(8)
  
        z = z + 1
        if len(lec) < 8:
          break
          
        (evt,timestamp) = struct.unpack(">LL", lec);
        timestamp = timestamp/1000
        if maxTimestamp < timestamp:
          maxTimestamp = timestamp
        if rowID == 0:
          minTimestamp = timestamp
  
        row = np.array([evt, timestamp - minTimestamp])
        aedatFile.append(row)
        rowID += 1
        position_pointer += 8
      maxTimestamp = maxTimestamp - minTimestamp
      minTimestamp = 0
      endTime = maxTimestamp
      aedatLength = rowID
    finally:
      f.close()
    return aedatFile
    
  record_on = None
  recorded_input_spiketimes = createSpikeSourceArray(loadAedat32(input_file))
  
  recorded_input_spiketimes_ON = [[0]]*len(recorded_input_spiketimes)
  for i in range(0, len(recorded_input_spiketimes)):
    if len(recorded_input_spiketimes[0:i]) % 2 == 0:
      recorded_input_spiketimes_ON[i/2] = recorded_input_spiketimes[i]
      
  recorded_input_spiketimes_OFF = [[0]]*len(recorded_input_spiketimes)
  for i in range(0, len(recorded_input_spiketimes)):
    if (len(recorded_input_spiketimes[0:i])-1) % 2 == 0:
      recorded_input_spiketimes_OFF[i/2] = recorded_input_spiketimes[i]
    
  record_on = recorded_input_spiketimes_ON[0:128**2]

  record_off = recorded_input_spiketimes_OFF[0:128**2]
  
  return[record_on, record_off]

def reorderDAVISdata(record, camera_resolution):
  events = []
  polarity = 1
  x = 0
  y = 0
  for indx in range(0, len(record)):
    for inter_indx in range(0, len(record[indx])):
      event = Event()

      event.x = x
      event.y = y
      event.timestamp = record[indx][inter_indx]
      event.polarity = polarity

      events.append(event)
    x += 1
    if (x == camera_resolution[0]):
      y += 1
      x = 0

  new_events = sorted(events, key=lambda x: x.timestamp)

  return new_events


# # input
# path_input = '/home/giulia/workspace/tde-iCubIIT/Stimuli/ThoerbenStimuli/icub_semd/recordings1/pattern_2_light_6_speed_3_number_1.aedat'
# N_AER_ADRESSES = 128**2*2
#
# # convert aedat file into spike array
#
# record_on, record_off = aedat_to_spikes(path_input, N_AER_ADRESSES)
