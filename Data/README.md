# sEMD-iCub

## Data

### Fake Stimuli

You can generate your own Fake stimuli. Have a look at the [README.md](Code/README.md) inside the Code folder. 

### Real-world data

You can find inside the folder [real-world-data.zip](real-world-data.zip) 8 different acquisition from the [ATIS cameras](https://ieeexplore.ieee.org/abstract/document/5981834) mounted on iCub.
The data represent the events stream of a black bar moving over a white canvas. 

Inside each folder (LR1, LR2, LR3, ..., LR8) you will find:

* data.log: events stream collected from the cameras via [YARP](https://www.yarp.it/) Network.

* data.log.txt: events stream converted in a txt file (from the second column: CPU time, channel, x, y, polarity, timestamp)

* info.log: information of the Yarp port sending the events. 

![lr](../images/events_stream.gif) 
