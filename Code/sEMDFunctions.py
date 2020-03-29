
from tqdm import tqdm

class neuron:

    def __init__(self):
        self.x=0
        self.y=0
        self.id=0
        self.timestamp=[]


def create_neurons(camera_resolution):
    id_neuron=0;
    NEURONS=[]
    for y in range(0, (camera_resolution[1])):
        for x in range(0, (camera_resolution[0])):
            neuron_tmp=neuron()

            neuron_tmp.x=x
            neuron_tmp.y=y
            neuron_tmp.id=id_neuron
            NEURONS.append(neuron_tmp)
            id_neuron+=1

    return NEURONS

def neurons_cell(events, RFs, neurons):
    for e in tqdm(events):
        for rf in RFs:
            if (rf.x == e.x and rf.y == e.y):
                neurons[rf.id].append(e.timestamp)
    return neurons