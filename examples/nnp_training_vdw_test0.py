# -*- coding: utf-8 -*-
"""
################ Work in Progress ################
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
from torchani import utils
import os
import math
import torch.utils.tensorboard
import tqdm

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ANIVModel(torch.nn.ModuleList):
    """ANIV model that compute properties from species and AEVs and also carries coordinates.
    Use to prepare for addition of vdw corrections

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
        super(ANIVModel, self).__init__(modules)
        self.reducer = reducer
        self.padding_fill = padding_fill

    def forward(self, species_aev_coordinates):
        species, aev, coordinates = species_aev_coordinates
        # species, aev = species_aev
        species_ = species.flatten()
        output_shape = list(species_.size())
        output_shape.append(2)
        present_species = utils.present_species(species)
        aev = aev.flatten(0, 1)

        # output = torch.zeros(output_shape, dtype=aev.dtype)
        output = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)
        output_c6 = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)

        for i in present_species:
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            res = self[i](input_)
            output.masked_scatter_(mask, res[:, 0].squeeze())
            output_c6.masked_scatter_(mask, res[:, 1].squeeze())
        aa_energies = output.view_as(species)
        c6 = output_c6.view_as(species)
        return species, self.reducer(aa_energies, dim=1), c6, coordinates


class VDWModule(torch.nn.Module):
    """The VDW Module that takes species and coordinates as input and outputs long-range VDW corrections.
    Arguments:
        species: list of species, (mol, atom)
        coordinates is a 3-dimension tensor (mol, atom, xyz)
    """
    def __init__(self):
        super(VDWModule, self).__init__()

    def vdw_func(self, species_, coords_, c6_):
        ################ Work in Progress ################
        species_mask = species_ <= -1
        table_masks = species_mask.unsqueeze(1) + species_mask.unsqueeze(2)
        # (mol, 1, atom, xyz) - (mol, atom, 1, xyz) => (mol, atom1, atom2, diff_xyz)
        dist = coords_.unsqueeze(1) - coords_.unsqueeze(2)
        # dist = torch.sqrt(torch.sum(dist ** 2, dim=-1) + 1e-6).unsqueeze(-1)
        # distances = dist.norm(2, -1)
        dist2 = torch.sum(dist ** 2, dim=-1)
        # .unsqueeze(-1)
        dist6 = dist2 ** 3
        # c6_ is regularized by sigmoid function to be in range of 0 to 10000.0 (unit Hartree/Ang^6)
        # Converstion factor to atomic unit (45.54 ~ 1.0/0.529^6)
        # suitable for HCNO and probably some other atoms/ions
        # (mol, 1, c6_atom) - (mol, c6_atom, 1) = > (mol, c6_atom1, c6_atom2)
        c6_ = 1000.0 / (1.0 + torch.exp(c6_))
        c6_ij = 2.0 * c6_.unsqueeze(1) * c6_.unsqueeze(2) / (c6_.unsqueeze(1) + c6_.unsqueeze(2))
        vdw_c6 = c6_ij / dist6
        vdw_c6 = torch.triu(vdw_c6, diagonal=1)
        # size = vdw_c6.shape[1]
        # vdw_c6[:, range(size), range(size)] = 0.0
        assert not (torch.isnan(vdw_c6)).any()
        assert not (vdw_c6 == float('inf')).all()
        # Fermi-type damping function using prefactor 20.0, rcut = 5.2
        # f_damp = 1.0 / (1.0 + math.exp(-20.0 * (math.sqrt(dist2) / 5.2 - 1.0)))
        # Modified Fermi-type damping function, rcut^2 = 25.0
        f_damp = 1.0 / (1.0 + torch.exp(-(dist2 - 25.0)))
        vdw_energies = -f_damp * vdw_c6
        vdw_energies.masked_fill_(table_masks, 0.0)
        vdw_energies = torch.sum(vdw_energies, dim=-1)
        vdw_energies = torch.sum(vdw_energies, dim=-1)
        return vdw_energies

    def forward(self, input_):
        """Compute van der waals interactions within each molecule
        Arguments:
            input_ (tuple): species, coordinates
            coordinates : (mol, atom, xyz)
        Returns:
            vdw_energies (tensor): (mol)
        """
        species, energies, c6, coords = input_
        vdw_energies = self.vdw_func(species, coords, c6)
        total_energies = energies + vdw_energies
        return species, total_energies

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

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()

dspath = os.path.join(path, '../dataset/ani-1x/sample.h5')

Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
num_species = 4
aevp_computer = torchani.aev.AEVPComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
# sae_file = os.path.join(path, '../torchani/resources/ani-1x_8x/sae_linfit.dat')
# energy_shifter = torchani.neurochem.load_sae(sae_file)
energy_shifter = torchani.utils.EnergyShifter(None)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')

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
    torch.nn.Linear(96, 2)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(384, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 2)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 2)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 2)
)

nn = ANIVModel([H_network, C_network, N_network, O_network])
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


nn.apply(init_params)

vdw = VDWModule()
###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
modelv = torch.nn.Sequential(aevp_computer, nn, vdw).to(device)

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
], lr=1e-3)

'''
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

###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
# SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################
# Train the model by minimizing the MSE loss, until validation RMSE no longer
# improves during a certain number of steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a threshold.
#
# We first read the checkpoint files to restart training. We use `latest.pt`
# to store current training state.
latest_checkpoint = 'nnp_vdw_01_latest.pt'

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    # SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    # SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


# helper function to convert energy unit from Hartree to kcal/mol
def hartree2kcal(x):
    return 627.509 * x


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    for batch_x, batch_y in validation:
        true_energies = batch_y['energies']
        predicted_energies = []
        for chunk_species, chunk_coordinates in batch_x:
            _, chunk_energies = modelv((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)
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
max_epochs = 200
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'nnp_vdw_01_best.pt'

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
    # SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, (batch_x, batch_y) in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):

        true_energies = batch_y['energies']
        predicted_energies = []
        num_atoms = []

        for chunk_species, chunk_coordinates in batch_x:
            num_atoms.append((chunk_species >= 0).to(true_energies.dtype).sum(dim=1))
            _, chunk_energies = modelv((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)

        num_atoms = torch.cat(num_atoms)
        predicted_energies = torch.cat(predicted_energies)
        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        # SGD.zero_grad()
        loss.backward()
        AdamW.step()
        # SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        # 'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        # 'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
