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


def convert_dual_coords(batch_y):
    """
    Args:
        batch_y: a dict('energies', tensor([#mols]); 'atomic': list[#chunk])
        chunk: a dict('species': tensor([#mols, #atoms]); 'coordinates': tensor([#mols, #atoms, 3])
    return:
        new dataset in batch form: a list[#chunks]
        chunk: a dict('species': tensor([#mols, #atoms]); 'coords1':, 'coords2':, 'ene1':, 'ene2':, 'ene_diff':)
    """
    energies = batch_y['energies']  # one dimension tensor
    idx_start = 0
    new_batch = []
    for chunk_dict in batch_y['atomic']:
        chunk_species = chunk_dict['species']
        chunk_coordinates = chunk_dict['coordinates']
        chunk_len = chunk_species.shape[0]
        chunk_energies = energies[idx_start:idx_start+chunk_len]
        new_chunk_dict = {}
        species_list = []
        coords1_list =[]
        coords2_list = []
        ene1_list =[]
        ene2_list =[]
        # ene_diff = []
        chunk_coordinates = chunk_coordinates.numpy()
        # coords1_array = np.empty(chunk_coordinates[0].shape, dtpye=np.float32)
        # coords2_array = np.empty(chunk_coordinates[0].shape, dtpye=np.float32)
        unique_species, loc, occur_count = np.unique(chunk_species.numpy(), return_inverse=True, return_counts=True, axis=0)
        # the current implementation assume small number of molecules has little conformers
        # it might cause issues when there are too many molecules with small number of conformers
        n_single_occur = len(np.asarray(occur_count == 1).nonzero()[0])
        n_double_occur = len(np.asarray(occur_count == 2).nonzero()[0])
        # n_triple_occur = len(np.asarray(occur_count == 3).nonzero()[0])
        n_multi_occur = len(np.asarray(occur_count > 3).nonzero()[0])
        max_extra_allowed = 0
        for multi_occur in np.asarray(occur_count > 3).nonzero()[0]:
            max_extra_allowed += int(multi_occur * (multi_occur - 1) / 2) - multi_occur
        n_reduced = n_single_occur + n_double_occur
        if max_extra_allowed < n_reduced:
            continue
        n_extra = n_reduced // n_multi_occur
        for i, chunk_unique_species in enumerate(unique_species):
            all_loc = np.asarray(loc == i).nonzero()[0]
            if len(all_loc) == 1:
                pass
            elif len(all_loc) < 4: # in case of 2 and 3
                for loc1_idx, loc2_idx in list(combinations(all_loc, 2)):
                    species_list.append(chunk_unique_species)
                    coords1_list.append(chunk_coordinates[loc1_idx])
                    coords2_list.append(chunk_coordinates[loc2_idx])
                    ene1_list.append(chunk_energies[loc1_idx])
                    ene2_list.append(chunk_energies[loc2_idx])
                    # ene_diff.append(chunk_energies[loc1_idx] - chunk_energies[loc2_idx])
            elif len(all_loc) > 3: # in case of 4 or more
                n_sample = len(all_loc)
                if n_reduced > 0:
                    extra_allowed = int(len(all_loc) * (len(all_loc) - 1) / 2) - len(all_loc)
                    extra_used = min(n_reduced, n_extra, extra_allowed)
                    n_reduced -= extra_used
                    n_sample += extra_used
                for loc1_idx, loc2_idx in random.sample(list(combinations(all_loc, 2)), n_sample):
                    species_list.append(chunk_unique_species)
                    coords1_list.append(chunk_coordinates[loc1_idx])
                    coords2_list.append(chunk_coordinates[loc2_idx])
                    ene1_list.append(chunk_energies[loc1_idx])
                    ene2_list.append(chunk_energies[loc2_idx])
                    # ene_diff.append(chunk_energies[loc1_idx] - chunk_energies[loc2_idx])

            new_chunk_dict['species'] = torch.LongTensor(species_list)
            new_chunk_dict['coords1'] = torch.FloatTensor(coords1_list)
            new_chunk_dict['coords2'] = torch.FloatTensor(coords2_list)
            new_chunk_dict['ene1'] = torch.FloatTensor(ene1_list)
            new_chunk_dict['ene2'] = torch.FloatTensor(ene2_list)
            # new_chunk_dict['ene_diff'] = torch.FloatTensor(ene_diff)
        new_batch.append(new_chunk_dict)
        idx_start += chunk_len

    return new_batch

class ANIDualModel(torch.nn.ModuleList):
    """ANI model that compute properties from species and two set of AEVs.

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
            ``(tensor, dim)->tensor``.
        padding_fill (float): The value to fill output of padding atoms.
            Padding values will participate in reducing, so this value should
            be appropriately chosen so that it has no effect on the result. For
            example, if the reducer is :func:`torch.sum`, then
            :attr:`padding_fill` should be 0, and if the reducer is
            :func:`torch.min`, then :attr:`padding_fill` should be
            :obj:`math.inf`.
    """

    def __init__(self, modules, reducer=torch.sum, padding_fill=0):
        super(ANIDualModel, self).__init__(modules)
        self.reducer = reducer
        self.padding_fill = padding_fill

    def forward(self, species_aevs):
        species, aev1, aev2 = species_aevs
        species_ = species.flatten()
        present_species = utils.present_species(species)
        aev1 = aev1.flatten(0, 1)
        aev2 = aev2.flatten(0, 1)

        output1 = torch.full_like(species_, self.padding_fill, dtype=aev1.dtype)
        output2 = torch.full_like(species_, self.padding_fill, dtype=aev2.dtype)

        for i in present_species:
            mask = (species_ == i)
            input1_ = aev1.index_select(0, mask.nonzero().squeeze())
            input2_ = aev2.index_select(0, mask.nonzero().squeeze())
            output1.masked_scatter_(mask, self[i](input1_).squeeze())
            output2.masked_scatter_(mask, self[i](input2_).squeeze())
        output1 = output1.view_as(species)
        output2 = output2.view_as(species)
        output1 = self.reducer(output1, dim=1)
        output2 = self.reducer(output2, dim=1)
        # delta_e = output1 - output2
        return species, output1, output2

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
aev_dual_computer = torchani.aev.AEVDualComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
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

batch_size = 2560

training, validation = torchani.data.load_ani_dataset(
    dspath, species_to_tensor, batch_size, rm_outlier=True, device=device,
    transform=[energy_shifter.subtract_from_dataset], split=[0.8, None])

print('Self atomic energies: ', energy_shifter.self_energies)

###############################################################################
# When iterating the dataset, we will get pairs of input and output
# ``(species_coordinates, properties)``, where ``species_coordinates`` is the
# input and ``properties`` is the output.
#
# ``species_coordinates`` is a list of species-coordinate pairs, with shape
# ``(N, Na)`` and ``(N, Na, 3)``. The reason for getting this type is, when
# loading the dataset and generating minibatches, the whole dataset are
# shuffled and each minibatch contains structures of molecules with a wide
# range of number of atoms. Molecules of different number of atoms are batched
# into single by padding. The way padding works is: adding ghost atoms, with
# species 'X', and do computations as if they were normal atoms. But when
# computing AEVs, atoms with species `X` would be ignored. To avoid computation
# wasting on padding atoms, minibatches are further splitted into chunks. Each
# chunk contains structures of molecules of similar size, which minimize the
# total number of padding atoms required to add. The input list
# ``species_coordinates`` contains chunks of that minibatch we are getting. The
# batching and chunking happens automatically, so the user does not need to
# worry how to construct chunks, but the user need to compute the energies for
# each chunk and concat them into single tensor.
#
# The output, i.e. ``properties`` is a dictionary holding each property. This
# allows us to extend TorchANI in the future to training forces and properties.
#
###############################################################################
# Now let's define atomic neural networks.

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

nn = ANIDualModel([H_network, C_network, N_network, O_network])
print(nn)

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


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

ani_1x_model = 'ani-1x_8x_model0.pt'

if os.path.isfile(ani_1x_model):
    checkpoint = torch.load(ani_1x_model)
    nn.load_state_dict(checkpoint)
    for i_module in range(4):
        for i_layer in range(0, 7, 2):
            if i_layer < 0:
                print('nn.module:', i_module, ' i_module.layer', i_layer, ' are fixed.')
                for param in nn[i_module][i_layer].parameters():
                    print(param.size())
                    param.requires_grad = False
            else:
                print('nn.module:', i_module, ' i_module.layer', i_layer, ' are free.')
                for param in nn[i_module][i_layer].parameters():
                    print(param.size())
else:
    nn.apply(init_params)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torch.nn.Sequential(aev_dual_computer, nn).to(device)

###############################################################################
# Now let's setup the optimizers. NeuroChem uses Adam with decoupled weight decay
# to updates the weights and Stochastic Gradient Descent (SGD) to update the biases.
# Moreover, we need to specify different weight decay rate for different layes.
#
# .. note::
#
#   The weight decay in `inputtrain.ipt`_ is named "l2", but it is actually not
#   L2 regularization. The confusion between L2 and weight decay is a common
#   mistake in deep learning.  See: `Decoupled Weight Decay Regularization`_
#   Also note that the weight decay only applies to weight in the training
#   of ANI models, not bias.
#
# .. _Decoupled Weight Decay Regularization:
#   https://arxiv.org/abs/1711.05101

AdamW = torchani.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].weight]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].weight]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].weight]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].weight]},
    {'params': [O_network[6].bias]},
])


###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)

###############################################################################
# Train the model by minimizing the MSE loss, until validation RMSE no longer
# improves during a certain number of steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a threshold.
#
# We first read the checkpoint files to restart training. We use `latest.pt`
# to store current training state.
latest_checkpoint = 'conf_latest.pt'

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


# helper function to convert energy unit from Hartree to kcal/mol
def hartree2kcal(x):
    return 627.509 * x

weight_ene_diff = 2.0

def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    for _, batch_y in validation:
        true_energies = []
        predicted_energies = []
        new_batch = convert_dual_coords(batch_y)
        for chunk_dict in new_batch:
            chunk_species = chunk_dict['species']
            chunk_coords1 = chunk_dict['coords1']
            chunk_coords2 = chunk_dict['coords2']
            chunk_ene1 = chunk_dict['ene1']
            chunk_ene2 = chunk_dict['ene2']
            chunk_ene_diff = chunk_ene1 - chunk_ene2

            _, chunk_energies1, chunk_energies2 = model((chunk_species, chunk_coords1, chunk_coords2))
            chunk_energies_diff = chunk_energies1 - chunk_energies2
            true_energies.extend((chunk_ene1, chunk_ene2, chunk_ene_diff))
            predicted_energies.extend((chunk_energies1, chunk_energies2, chunk_energies_diff))
        true_energies = torch.cat(true_energies)
        predicted_energies = torch.cat(predicted_energies)
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcal(math.sqrt(total_mse / count))


###############################################################################
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# Finally, we come to the training loop.
#
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 1200
early_stopping_learning_rate = 1.0E-6
best_model_checkpoint = 'conf_best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, (batch_x, batch_y) in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):

        true_energies = []
        predicted_energies = []
        num_atoms = []
        new_batch = convert_dual_coords(batch_y)

        for chunk_dict in new_batch:
            chunk_species = chunk_dict['species']
            chunk_coords1 = chunk_dict['coords1']
            chunk_coords2 = chunk_dict['coords2']
            chunk_ene1 = chunk_dict['ene1']
            chunk_ene2 = chunk_dict['ene2']
            chunk_ene_diff = chunk_ene1 - chunk_ene2

            num_atoms.append((chunk_species >= 0).to(chunk_ene1.dtype).sum(dim=1)) # for ene1
            num_atoms.append((chunk_species >= 0).to(chunk_ene1.dtype).sum(dim=1)) # for ene2
            num_atoms.append((chunk_species >= 0).to(chunk_ene1.dtype).sum(dim=1)) # for ene_diff

            _, chunk_energies1, chunk_energies2 = model((chunk_species, chunk_coords1, chunk_coords2))
            chunk_energies_diff = chunk_energies1 - chunk_energies2
            true_energies.extend((chunk_ene1, chunk_ene2, chunk_ene_diff))
            predicted_energies.extend((chunk_energies1, chunk_energies2, chunk_energies_diff))

        true_energies = torch.cat(true_energies)
        predicted_energies = torch.cat(predicted_energies)

        num_atoms = torch.cat(num_atoms)
        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        loss.backward()
        AdamW.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
    }, latest_checkpoint)
