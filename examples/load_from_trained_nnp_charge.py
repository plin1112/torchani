# -*- coding: utf-8 -*-
"""
.. _charge-training-example:

Train Neural Network Charges To Charges only
==========================================================

We have seen how to train a neural network potential by manually writing
training loop in :ref:`training-example`. This tutorial shows how to modify
that script to train to charges.
"""

###############################################################################
# Most part of the script are the same as :ref:`training-example`, we will omit
# the comments for these parts. Please refer to :ref:`training-example` for more
# information

import torch
import torchani
import os
import torchani.utils

class ANICModel(torch.nn.ModuleList):
    """ANIC model that compute properties from species and AEVs.

    Different atom types might have different modules, when computing
    properties, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular properties.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
        reducer (:class:`collections.abc.Callable`): The callable that reduce
            atomic outputs into molecular outputs. It must have signature
            ``(tensor, dim)->tensor``. # Not used in ANICModel
        padding_fill (float): The value to fill output of padding atoms.
            Padding values will participate in reducing, so this value should
            be appropriately chosen so that it has no effect on the result. For
            example, if the reducer is :func:`torch.sum`, then
            :attr:`padding_fill` should be 0, and if the reducer is
            :func:`torch.min`, then :attr:`padding_fill` should be
            :obj:`math.inf`. # Not used in ANICModel
    """

    def __init__(self, modules, padding_fill=0):
        super(ANICModel, self).__init__(modules)
        self.padding_fill = padding_fill

    def forward(self, species_aev):
        species, aev = species_aev
        species_ = species.flatten()
        present_species = torchani.utils.present_species(species)
        aev = aev.flatten(0, 1)

        output = torch.full_like(species_, self.padding_fill,
                                 dtype=aev.dtype)
        for i in present_species:
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            output.masked_scatter_(mask, self[i](input_).squeeze())
        output = output.view_as(species)
        return species, output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    Rcr = 5.2000e+00
    Rca = 3.5000e+00
    EtaR = torch.tensor([1.6000000e+01], device=device)
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
    Zeta = torch.tensor([3.2000000e+01], device=device)
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
    EtaA = torch.tensor([8.0000000e+00], device=device)
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
    num_species = 4
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    energy_shifter = torchani.utils.EnergyShifter(None)
    species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath = os.path.join(path, '../dataset/COMP6/COMP6v1/DrugBank/gdb13_13_test1000_mod.h5')

# batch_size = 2560
batch_size = 256

###############################################################################
# The code to create the dataset is a bit different: we need to manually
# specify that ``atomic_properties=['cm5']`` so that charges will be read
# from hdf5 files.

training, validation = torchani.data.load_ani_dataset(
    dspath, species_to_tensor, batch_size, rm_outlier=True,
    device=device, atomic_properties=['cm5'],
    transform=[energy_shifter.subtract_from_dataset], split=[0.0, None])

print('Self atomic energies: ', energy_shifter.self_energies)

###############################################################################
# When iterating the dataset, we will get pairs of input and output
# ``(species_coordinates, properties)``, in this case, ``properties`` would
# contain a key ``'atomic'`` where ``properties['atomic']`` is a list of dict
# containing forces:

data = validation[0]
properties = data[1]
atomic_properties = properties['atomic']
print(type(atomic_properties))
print(list(atomic_properties[0].keys()))

###############################################################################
# Due to padding, part of the charges might be 0
print(atomic_properties[0]['cm5'][0])

###############################################################################
# The code to define networks, optimizers, are mostly the same

H_network = torch.nn.Sequential(
    torch.nn.Linear(384, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(384, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = ANICModel([H_network, C_network, N_network, O_network])
# print(nn)

###############################################################################
# Initialize the weights and biases.
#
# .. note::
#   Pytorch default initialization for the weights and biases in linear layers
#   is Kaiming uniform. See: `TORCH.NN.MODULES.LINEAR`_
#   We initialize the weights similarly but from the normal distribution.
#   The biases were initialized to zero.
#
# .. _TORCH.NN.MODULES.LINEAR:
#   https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

@torch.no_grad()
def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


nn.apply(init_params)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
# model = torch.nn.Sequential(aev_computer, nn).to(device)
modelc = torch.nn.Sequential(aev_computer, nn).to(device)
###############################################################################
# Here we will use Adam with weight decay for the weights and Stochastic Gradient
# Descent for biases.
'''
AdamW = torchani.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
])

SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)
'''

# AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
# SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################
# This part of the code is also the same
# latest_checkpoint = 'charge-training-latest.pt'
best_model_checkpoint = 'charge-training-best.pt'

###############################################################################
# Resume training from previously saved checkpoints:
# if os.path.isfile(latest_checkpoint):
#     checkpoint = torch.load(latest_checkpoint)
#     nn.load_state_dict(checkpoint['nn'])
#     AdamW.load_state_dict(checkpoint['AdamW'])
#     SGD.load_state_dict(checkpoint['SGD'])
#     AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
#     SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

if os.path.isfile(best_model_checkpoint):
    checkpoint = torch.load(best_model_checkpoint)
    nn.load_state_dict(checkpoint)

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint

# helper function to convert energy unit from Hartree to kcal/mol
# def hartree2kcal(x):
#     return 627.509 * x

################################################################################
# Use trained network to calculate the charges for each molecule/conformation
fout = open('comparison_cm5_gdb13_13_test1000_mod.txt', 'w+')

for _, batch_x in validation:
    for chunk in batch_x['atomic']:
        chunk_species = chunk['species']
        chunk_coordinates = chunk['coordinates']
        chunk_true_charges = chunk['cm5']
        true_charges = torch.flatten(chunk_true_charges)
        _, chunk_charges = modelc((chunk_species, chunk_coordinates))
        predicted_charges = torch.flatten(chunk_charges)
        # print(chunk['path'])
        species_list = [a.item() for a in chunk_species.flatten()]
        true_charges_list = [b.item() for b in chunk_true_charges.flatten()]
        chunk_charges_list = [c.item() for c in chunk_charges.flatten()]
        for a, b, c, in zip(species_list, true_charges_list, chunk_charges_list):
            if a >= 0:
                fout.write('%-5s %10.5f %10.5f\n' % (a, b, c))

fout.close()