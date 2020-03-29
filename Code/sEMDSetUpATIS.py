import socket
import spynnaker8 as sim
import numpy as np
#import logging
import matplotlib.pyplot as plt

from pacman.model.constraints.key_allocator_constraints import FixedKeyAndMaskConstraint
from pacman.model.graphs.application import ApplicationSpiNNakerLinkVertex
from pacman.model.routing_info import BaseKeyAndMask
from spinn_front_end_common.abstract_models.abstract_provides_n_keys_for_partition import AbstractProvidesNKeysForPartition
from spinn_front_end_common.abstract_models.abstract_provides_outgoing_partition_constraints import AbstractProvidesOutgoingPartitionConstraints
from spinn_utilities.overrides import overrides
from spinn_front_end_common.abstract_models.abstract_provides_incoming_partition_constraints import AbstractProvidesIncomingPartitionConstraints
from pacman.executor.injection_decorator import inject_items
from pacman.operations.routing_info_allocator_algorithms.malloc_based_routing_allocator.utils import get_possible_masks
from spinn_front_end_common.utility_models.command_sender_machine_vertex import CommandSenderMachineVertex

from spinn_front_end_common.abstract_models \
    import AbstractSendMeMulticastCommandsVertex
from spinn_front_end_common.utility_models.multi_cast_command \
    import MultiCastCommand


from pyNN.utility import Timer
from pyNN.utility.plotting import Figure, Panel
from pyNN.random import RandomDistribution, NumpyRNG

from spynnaker.pyNN.models.neuron.plasticity.stdp.common import plasticity_helpers

NUM_NEUR_IN = 1024  # 2x240x304 mask -> 0xFFFE0000
MASK_IN = 0xFFFFFC00 #0xFFFFFC00
NUM_NEUR_OUT = 1024
#MASK_OUT =0xFFFFFC00

class Weights:
    pass

    sptc = 0.3  # was 0.6
    semd = 0.8  # 0.1
delay = 1.0
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 32)
# neuron parameters
cell_params_semd = {'cm'        : 0.25,
                    'i_offset'  : 0,                        # offset current
                    'tau_m'     : 10,                       # membrane potential time constant
                    'tau_refrac': 1,                        # refractory period time constant
                    'tau_syn_E' : 20 ,                      # excitatory current time constant
                    'tau_syn_I' : 20 ,                      # inhibitory current time constant
                    'v_reset'   : -85,                      # reset potential
                    'v_rest'    : -60,                      # resting potential
                    'v_thresh'  : -50                       # spiking threshold
                              }





class ICUBInputVertex(
        ApplicationSpiNNakerLinkVertex,
        AbstractProvidesOutgoingPartitionConstraints,
        AbstractProvidesIncomingPartitionConstraints,
        AbstractSendMeMulticastCommandsVertex):

    def __init__(self, spinnaker_link_id, board_address=None,
                 constraints=None, label=None):

        ApplicationSpiNNakerLinkVertex.__init__(
            self, n_atoms=NUM_NEUR_IN, spinnaker_link_id=spinnaker_link_id,
            board_address=board_address, label=label)

        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        AbstractSendMeMulticastCommandsVertex.__init__(self)

    @overrides(AbstractProvidesOutgoingPartitionConstraints.
               get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [FixedKeyAndMaskConstraint(
            keys_and_masks=[BaseKeyAndMask(
                base_key=0, #upper part of the key,
                mask=MASK_IN)])]
                #keys, i.e. neuron addresses of the input population that sits in the ICUB vertex,

    @inject_items({"graph_mapper": "MemoryGraphMapper"})
    @overrides(AbstractProvidesIncomingPartitionConstraints.
               get_incoming_partition_constraints,
               additional_arguments=["graph_mapper"])
    def get_incoming_partition_constraints(self, partition, graph_mapper):
        if isinstance(partition.pre_vertex, CommandSenderMachineVertex):
            return []
        index = graph_mapper.get_machine_vertex_index(partition.pre_vertex)
        vertex_slice = graph_mapper.get_slice(partition.pre_vertex)
        mask = get_possible_masks(vertex_slice.n_atoms)[0]
        key = (0x1000 + index) << 16
        return [FixedKeyAndMaskConstraint(
            keys_and_masks=[BaseKeyAndMask(key, mask)])]

    @property
    @overrides(AbstractSendMeMulticastCommandsVertex.start_resume_commands)
    def start_resume_commands(self):
        return [MultiCastCommand(
            key=0x80000000, payload=0, repeat=5, delay_between_repeats=100)]

    @property
    @overrides(AbstractSendMeMulticastCommandsVertex.pause_stop_commands)
    def pause_stop_commands(self):
        return [MultiCastCommand(
            key=0x40000000, payload=0, repeat=5, delay_between_repeats=100)]

    @property
    @overrides(AbstractSendMeMulticastCommandsVertex.timed_commands)
    def timed_commands(self):
        return []


sim.setup(timestep=1.0)
sim.set_number_of_neurons_per_core(sim.extra_models.IF_curr_exp_sEMD, 100)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 32)


# set up populations,
pop = sim.Population(None, ICUBInputVertex(spinnaker_link_id=0), label='pop_in')
checkIn = sim.Population(NUM_NEUR_OUT, sim.IF_curr_exp(), label='neuron_pop')
sim.Projection(pop, checkIn, sim.OneToOneConnector(), sim.StaticSynapse(weight=20.0))
neuron_pop_lr = sim.Population(NUM_NEUR_OUT, sim.extra_models.IF_curr_exp_sEMD, cell_params_semd, label='neuron_pop')
neuron_pop_rl = sim.Population(NUM_NEUR_OUT, sim.extra_models.IF_curr_exp_sEMD, cell_params_semd, label='neuron_pop')
neuron_pop_tb = sim.Population(NUM_NEUR_OUT, sim.extra_models.IF_curr_exp_sEMD, cell_params_semd, label='neuron_pop')
neuron_pop_bt = sim.Population(NUM_NEUR_OUT, sim.extra_models.IF_curr_exp_sEMD, cell_params_semd, label='neuron_pop')

# neural connections
semd_lr_fac = [(i, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
semd_lr_trig = [(i+1, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
semd_rl_fac = [(i+1, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
semd_rl_trig = [(i, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
semd_tb_fac = [(i, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
semd_tb_trig = [(i+30, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
semd_bt_fac = [(i+32, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
semd_bt_trig = [(i, i, Weights.semd, delay) for i in range(NUM_NEUR_OUT)]
#neural projections    ,

sim.Projection(checkIn, neuron_pop_lr, sim.FromListConnector(semd_lr_fac), sim.StaticSynapse(), receptor_type='excitatory')
sim.Projection(checkIn, neuron_pop_lr, sim.FromListConnector(semd_lr_trig), sim.StaticSynapse(), receptor_type='excitatory2')
sim.Projection(checkIn, neuron_pop_rl, sim.FromListConnector(semd_rl_fac), sim.StaticSynapse(), receptor_type='excitatory')
sim.Projection(checkIn, neuron_pop_rl, sim.FromListConnector(semd_rl_trig), sim.StaticSynapse(), receptor_type='excitatory2')
sim.Projection(checkIn, neuron_pop_tb, sim.FromListConnector(semd_tb_fac), sim.StaticSynapse(), receptor_type='excitatory')
sim.Projection(checkIn, neuron_pop_tb, sim.FromListConnector(semd_tb_trig), sim.StaticSynapse(), receptor_type='excitatory2')
sim.Projection(checkIn, neuron_pop_bt, sim.FromListConnector(semd_bt_fac), sim.StaticSynapse(), receptor_type='excitatory')
sim.Projection(checkIn, neuron_pop_bt, sim.FromListConnector(semd_bt_trig), sim.StaticSynapse(), receptor_type='excitatory2')


# pop_out = sim.Population(None, ICUBOutputVertex(spinnaker_link_id=0), label='pop_out')

sim.external_devices.activate_live_output_to(checkIn, pop)

#recordings and simulations,
checkIn.record("spikes")
neuron_pop_lr.record("spikes")
neuron_pop_rl.record("spikes")
neuron_pop_tb.record("spikes")
neuron_pop_bt.record("spikes")

simtime = 30000 #ms,
sim.run(simtime)

# continuous run until key press
# remember: do NOT record when running in this mode
# sim.external_devices.run_forever()
# raw_input('Press enter to stop')

exc_spikes = checkIn.get_data("spikes")
rl = neuron_pop_rl.get_data("spikes")
lr = neuron_pop_lr.get_data("spikes")
bt = neuron_pop_bt.get_data("spikes")
tb = neuron_pop_tb.get_data("spikes")

Figure(
#     raster plot of the neuron_pop
    Panel(exc_spikes.segments[0].spiketrains, xlabel="Time/ms", xticks=True,
          yticks=True, markersize=0.2, xlim=(0, simtime)),
    Panel(rl.segments[0].spiketrains, xlabel="Time/ms", ylabel="rl",xticks=True,
          yticks=True, markersize=0.2, xlim=(0, simtime)),
    Panel(lr.segments[0].spiketrains, xlabel="Time/ms",ylabel="lr", xticks=True,
          yticks=True, markersize=0.2, xlim=(0, simtime)),
    Panel(tb.segments[0].spiketrains, xlabel="Time/ms", ylabel="rl", xticks=True,
          yticks=True, markersize=0.2, xlim=(0, simtime)),
    Panel(bt.segments[0].spiketrains, xlabel="Time/ms", ylabel="lr", xticks=True,
          yticks=True, markersize=0.2, xlim=(0, simtime)),
    title="neuron_pop: spikes"
)
plt.show()

sim.end()

print("finished")
