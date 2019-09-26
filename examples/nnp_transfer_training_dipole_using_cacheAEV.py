# -*- coding: utf-8 -*-
"""
################ Work in Progress ################
.. _charge-training-example:

Train Neural Network Charges To Charges only using transfer learning
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
import math
import torch.utils.tensorboard
import tqdm
import torchani.utils

class ANICModel(torch.nn.ModuleList):
    """ANIC model that compute charge properties from species and AEVs.

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

def dipole_moment(coords_charges):
    """
    Calculates dipole moment of the molecule.
    coords_charges: tuple of atomic coordinates and point charges
    coords is a 3-dimension tensor (mol, atom, xyz)
    charges is a 2-dimension tensor (mol, atom)
    """
    toDebye = 4.80320425
    coords, charges = coords_charges
    charges = charges.unsqueeze(-1)
    dipoles = coords * charges
    dipoles = torch.sum(dipoles, dim=1)
    return dipoles * toDebye

class ANIDModel(torch.nn.ModuleList):
    """ANID model that compute dipole properties from species, coordinates and AEVs.

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
        super(ANIDModel, self).__init__(modules)
        self.padding_fill = padding_fill

    def forward(self, species_aev, coordinates):
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
        total_charges = torch.sum(output, dim=1)
        num_atoms = (species >= 0).to(total_charges.dtype).sum(dim=1)
        excess_charge = - total_charges / num_atoms
        excess_charge = excess_charge.unsqueeze(-1)
        output += excess_charge
        mask2 = (species < 0)
        set_zero = torch.full_like(species_, self.padding_fill, dtype=total_charges.dtype)
        output.masked_scatter_(mask2, set_zero)
        # total_charges = torch.sum(output, dim=1) # return the residual total_charges before correction
        return species, dipole_moment((coordinates, output)), total_charges

class DipoleModule(torch.nn.Module):
    r"""The Dipole Module that takes coordinates and charges as input and outputs dipoles of molecules.

    Arguments:
        species: list of species, (not used)
        coordinates is a 3-dimension tensor (mol, atom, xyz)
        charges is a 2-dimension tensor (mol, atom)
    """

    def __init__(self):
        super(DipoleModule, self).__init__()

    def forward(self, input_):
        """Compute Dipole moments of molecules

        Arguments:
            input_ (tuple): species, coordinates, charges
        Returns:
            dipole_moment (tensor): (mol, dipole_xyz)
        """
        species, coords, charges = input_
        charges = charges.unsqueeze(-1)
        dipoles = coords * charges
        dipoles = torch.sum(dipoles, dim=1)
        return dipoles


class VDWModule(torch.nn.Module):
    r"""The VDW Module that takes species and coordinates as input and outputs long-range VDW corrections.

    Arguments:
        species: list of species, (mol, atom)
        coordinates is a 3-dimension tensor (mol, atom, xyz)
    """

    def __init__(self):
        super(VDWModule, self).__init__()

    def vdw_func(self, species_, coords_):
        ################ Work in Progress ################
        energies = None
        return energies

    def forward(self, input_):
        """Compute van der waals interactions within each molecule

        Arguments:
            input_ (tuple): species, coordinates
        Returns:
            vdw_energies (tensor): (mol)
        """
        species, coords = input_
        vdw_energies = self.vdw_func(species, coords)
        return vdw_energies

class ELECModule(torch.nn.Module):
    r"""The ELEC Module that takes species and coordinates as input and outputs long-range electrostatic corrections.

    Arguments:
        species: list of species, (not used)
        coordinates is a 3-dimension tensor (mol, atom, xyz)
        charges is a 2-dimension tensor (mol, atom)
    """

    def __init__(self):
        super(ELECModule, self).__init__()

    def elec_func(self, coords_, charges_):
        ################ Work in Progress ################
        energies = None
        return energies

    def forward(self, input_):
        """Compute long-range electrostatic correction of molecules

        Arguments:
            input_ (tuple): species, coordinates, charges
        Returns:
            elec_energies (tensor): (mol)
        """
        species, coords, charges = input_
        elec_energies = self.elec_func(coords, charges)
        return elec_energies


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# dspath = os.path.join(path, '../dataset/COMP6/COMP6v1/DrugBank/drugbank_testset_mod2.h5')

# batch_size = 2560
batch_size = 256

# checkpoint file for best model and latest model
best_model_checkpoint = 'dipole-transfer-training-best0-4.pt'
latest_checkpoint = 'dipole-transfer-training-latest0-4.pt'

# save existing model parameters into checkpoint file format
const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
consts = torchani.neurochem.Constants(const_file)
model_dir = os.path.join(path, '../torchani/resources/ani-1x_8x/train0/networks')  # noqa: E501
model = torchani.neurochem.load_model(consts.species, model_dir)

ani_1x_model = 'ani-1x_t0_model0.pt'
if not os.path.exists(ani_1x_model):
    torch.save(model.state_dict(), ani_1x_model)

max_epochs = 300
early_stopping_learning_rate = 1.0E-5
dipole_coefficient = 1.0  # controls the importance of dipole loss
total_charge_coefficient = 10000.0  # controls the importance of sum of charge loss

###############################################################################
# The code to create the dataset is a bit different: we need to manually
# specify that ``atomic_properties=['cm5']`` so that charges will be read
# from hdf5 files.
training_cache = './nnp_dipole_training_cache'
validation_cache = './nnp_dipole_validation_cache'

training_generator = torchani.data.AEVPCacheLoader(training_cache, selection=['dipole', 'coordinates'])
validation_generator = torchani.data.AEVPCacheLoader(validation_cache, selection=['dipole', 'coordinates'])

total_training_batches = sum(1 for _ in training_generator)

###############################################################################
# When iterating the dataset, we will get pairs of input and output
# ``(species_coordinates, properties)``, in this case, ``properties`` would
# contain a key ``'atomic'`` where ``properties['atomic']`` is a list of dict
# containing forces:

###############################################################################
# Due to padding, part of the charges might be 0

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

nn = ANIDModel([H_network, C_network, N_network, O_network]).to(device)
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

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

if os.path.isfile(ani_1x_model):
    checkpoint = torch.load(ani_1x_model)
    nn.load_state_dict(checkpoint)
    for i_module in range(4):
        for i_layer in range(0, 7, 2):
            if i_layer < 0:
                # print('nn.module:', i_module, ' i_module.layer', i_layer, ' are fixed.')
                for param in nn[i_module][i_layer].parameters():
                    # print(param.size())
                    param.requires_grad = False
            else:
                # print('nn.module:', i_module, ' i_module.layer', i_layer, ' are free.')
                for param in nn[i_module][i_layer].parameters():
                    pass
                    # print(param.size())
else:
    nn.apply(init_params)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
# model = torch.nn.Sequential(aev_computer, nn).to(device)
# modelc = torch.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Here we will use Adam with weight decay for the weights and Stochastic Gradient
# Descent for biases.

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

AdamB = torchani.optim.AdamW([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias], 'weight_decay': 0.00001},
    {'params': [H_network[4].bias], 'weight_decay': 0.000001},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias], 'weight_decay': 0.00001},
    {'params': [C_network[4].bias], 'weight_decay': 0.000001},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias], 'weight_decay': 0.00001},
    {'params': [N_network[4].bias], 'weight_decay': 0.000001},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias], 'weight_decay': 0.00001},
    {'params': [O_network[4].bias], 'weight_decay': 0.000001},
    {'params': [O_network[6].bias]},
])

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

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
AdamB_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamB, factor=0.5, patience=100, threshold=0)
# SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################
# This part of the code is also the same

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    AdamB.load_state_dict(checkpoint['AdamB'])
    # SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    AdamB_scheduler.load_state_dict(checkpoint['AdamB_scheduler'])
    # SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint

# helper function to convert energy unit from Hartree to kcal/mol
def hartree2kcal(x):
    return 627.509 * x

@torch.no_grad()
def validate():
    # run validation
    mse = torch.nn.MSELoss(reduction='none')
    total_mse = 0.0
    mol_charges_mse = 0.0
    count = 0

    for valid_batch, output, valid_labels in validation_generator:
        # species_aevs, output obtained from validation_generator
        predicted_dipoles = torch.Tensor().to(device)
        true_dipoles = output['dipole']
        num_mols = true_dipoles.shape[0]
        for chunk, chunk_labels in zip(valid_batch, valid_labels):
            # chunk_species, chunk_aevs = chunk
            # chunk_true_dipoles = chunk_labels['dipole']
            chunk_coords = chunk_labels['coordinates']
            _, chunk_dipoles, total_charges = nn(chunk, chunk_coords)
            # true_dipoles = torch.cat((true_dipoles, torch.flatten(chunk_true_dipoles)))
            predicted_dipoles = torch.cat((predicted_dipoles, chunk_dipoles))
            # chunk_num_mols = torch.tensor(chunk_dipoles.shape[0]).to(num_mols.dtype)
            # num_mols += chunk_num_mols

        count += num_mols
        mol_charges = torch.zeros(total_charges.shape, dtype=total_charges.dtype)
        mol_charges_mse += mse(mol_charges, total_charges).sum()
        total_mse += mse(true_dipoles, predicted_dipoles).sum()
    total_mse = total_mse / count
    mol_charges_mse = mol_charges_mse / count
    print(total_mse, mol_charges_mse)
    return math.sqrt(total_mse)

###############################################################################
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# In the training loop, we need to compute force, and loss for forces
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)

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
    AdamB_scheduler.step(rmse)
    # SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    # Besides being stored in x, species and coordinates are also stored in y.
    # So here, for simplicity, we just ignore the x and use y for everything.
    i = 0
    for train_batch, output, train_labels in tqdm.tqdm(
        training_generator,
        total=total_training_batches,
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):

        predicted_dipoles = torch.Tensor().to(device)
        batch_total_charges = torch.Tensor().to(device)
        true_dipoles = output['dipole']
        num_mols = true_dipoles.shape[0]
        dipole_loss = 0.0
        mol_charges_loss = 0.0
        i += 1

        for chunk, chunk_labels in zip(train_batch, train_labels):
            # chunk_species, chunk_aevs = chunk
            # chunk_true_dipoles = chunk_labels['dipole']
            chunk_coords = chunk_labels['coordinates']
            _, chunk_dipoles, chunk_total_charges = nn(chunk, chunk_coords)
            # true_dipoles = torch.cat((true_dipoles, torch.flatten(chunk_true_dipoles)))
            predicted_dipoles = torch.cat((predicted_dipoles, chunk_dipoles))
            batch_total_charges = torch.cat((batch_total_charges, chunk_total_charges))
            # chunk_num_mols = torch.tensor(chunk_dipoles.shape[0]).to(num_mols.dtype)
            # num_mols += chunk_num_mols

        mol_charges = torch.zeros(batch_total_charges.shape)
        mol_charges_loss += mse(mol_charges, batch_total_charges).sum() / num_mols
        dipole_loss += mse(true_dipoles, predicted_dipoles).sum() / num_mols

        # Now the total loss is charge loss
        # dipole_loss = torch.cat(dipole_loss).mean()
        loss = dipole_coefficient * dipole_loss  # + total_charge_coefficient * mol_charges_loss

        AdamW.zero_grad()
        AdamB.zero_grad()
        # SGD.zero_grad()
        loss.backward()
        AdamW.step()
        AdamB.step()
        # SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * total_training_batches + i)
        tensorboard.add_scalar('batch_charge_loss', mol_charges_loss, AdamW_scheduler.last_epoch * total_training_batches + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamB': AdamB.state_dict(),
        # 'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'AdamB_scheduler': AdamB_scheduler.state_dict(),
        # 'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)

final_rmse = validate()
print('Final RMSE:', final_rmse, 'after epoch', AdamW_scheduler.last_epoch)
