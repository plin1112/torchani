# -*- coding: utf-8 -*-
"""
.. _training-example:

Train Your Own Neural Network Potential
=======================================

This example shows how to use TorchANI to train a neural network potential
with the setup identical to NeuroChem. We will use the same configuration as
specified in `inputtrain.ipt`_

.. _`inputtrain.ipt`:
    https://github.com/aiqm/torchani/blob/master/torchani/resources/ani-1x_8x/inputtrain.ipt

.. note::
    TorchANI provide tools to run NeuroChem training config file `inputtrain.ipt`.
    See: :ref:`neurochem-training`.
"""

###############################################################################
# To begin with, let's first import the modules and setup devices we will use:

import torch
import torchani
import os
import math
from torchani import utils
import torch.utils.tensorboard
import tqdm
import numpy as np
from itertools import combinations
import random

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Now let's setup constants and construct an AEV computer. These numbers could
# be found in `rHCNO-5.2R_16-3.5A_a4-8.params`
# The atomic self energies given in `sae_linfit.dat`_ are computed from ANI-1x
# dataset. These constants can be calculated for any given dataset if ``None``
# is provided as an argument to the object of :class:`EnergyShifter` class.
#
# .. note::
#
#   Besides defining these hyperparameters programmatically,
#   :mod:`torchani.neurochem` provide tools to read them from file. See also
#   :ref:`training-example-ignite` for an example of usage.
#
# .. _rHCNO-5.2R_16-3.5A_a4-8.params:
#   https://github.com/aiqm/torchani/blob/master/torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params
# .. _sae_linfit.dat:
#   https://github.com/aiqm/torchani/blob/master/torchani/resources/ani-1x_8x/sae_linfit.dat

Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
num_species = 4
aev_computer = torchani.aev.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
# energy_shifter = torchani.utils.EnergyShifter(None)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts("HCNO")

###############################################################################
# Now let's setup datasets. These paths assumes the user run this script under
# the ``examples`` directory of TorchANI's repository. If you download this
# script, you should manually set the path of these files in your system before
# this script can run successfully.
#
# Also note that we need to subtracting energies by the self energies of all
# atoms for each molecule. This makes the range of energies in a reasonable
# range. The second argument defines how to convert species as a list of string
# to tensor, that is, for all supported chemical symbols, which is correspond to
# ``0``, which correspond to ``1``, etc.

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
# dspath = os.path.join(path, '../dataset/ani1x-20191001_wb97x_dz_dipole.h5')
dspath = os.path.join(path, '../dataset/ani-1x/ani_al-901_validation.h5')

# Now let's read self energies and construct energy shifter.
sae_file = os.path.join(path, '../torchani/resources/ani-1x_8x/sae_linfit.dat')  # noqa: E501
energy_shifter = torchani.neurochem.load_sae(sae_file)

print('Self atomic energies: ', energy_shifter.self_energies)

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

nn = torchani.nn.ANIModel([H_network, C_network, N_network, O_network])
print(nn)

ani_1x_model = 'ani-1x_8x_model0.pt'
if os.path.isfile(ani_1x_model):
    checkpoint = torch.load(ani_1x_model)
    nn.load_state_dict(checkpoint)
else:
    print('ani_1x_model not found!')
    exit()

model1 = torch.nn.Sequential(aev_computer, nn).to(device)

best_model_checkpoint = 'conf_best.pt'
if os.path.isfile(best_model_checkpoint):
    checkpoint = torch.load(best_model_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
else:
    print('best_model_checkpoint not found!')
    exit()

model2 = torch.nn.Sequential(aev_computer, nn).to(device)

def hartree2kcal(x):
    return 627.509 * x
