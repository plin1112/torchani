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
import math
import torch.utils.tensorboard
import tqdm
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
dspath = os.path.join(path, '../dataset/COMP6/COMP6v1/DrugBank/drugbank_testset_mod.h5')

# batch_size = 2560
batch_size = 256

###############################################################################
# The code to create the dataset is a bit different: we need to manually
# specify that ``atomic_properties=['cm5']`` so that charges will be read
# from hdf5 files.

training, validation = torchani.data.load_ani_dataset(
    dspath, species_to_tensor, batch_size, rm_outlier=True,
    device=device, atomic_properties=['cm5'],
    transform=[energy_shifter.subtract_from_dataset], split=[0.8, None])

print('Self atomic energies: ', energy_shifter.self_energies)

###############################################################################
# When iterating the dataset, we will get pairs of input and output
# ``(species_coordinates, properties)``, in this case, ``properties`` would
# contain a key ``'atomic'`` where ``properties['atomic']`` is a list of dict
# containing forces:

data = training[0]
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

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
# model = torch.nn.Sequential(aev_computer, nn).to(device)
modelc = torch.nn.Sequential(aev_computer, nn).to(device)
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

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################
# This part of the code is also the same
latest_checkpoint = 'charge-training-latest.pt'

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


# helper function to convert energy unit from Hartree to kcal/mol
def hartree2kcal(x):
    return 627.509 * x

def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='none')
    total_mse = 0.0
    count = 0
    '''
    for batch_x, batch_y in validation:
        true_energies = batch_y['energies']
        predicted_energies = []
        for chunk_species, chunk_coordinates in batch_x:
            _, chunk_energies = model((chunk_species, chunk_coordinates))
            predicted_energies.append(chunk_energies)
        predicted_energies = torch.cat(predicted_energies)
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcal(math.sqrt(total_mse / count)) 
    '''
    for _, batch_x in validation:
        predicted_charges = torch.Tensor()
        true_charges = torch.Tensor()
        num_atoms = 0.0
        for chunk in batch_x['atomic']:
            chunk_species = chunk['species']
            chunk_coordinates = chunk['coordinates']
            chunk_true_charges = chunk['cm5']
            true_charges = torch.cat((true_charges, torch.flatten(chunk_true_charges)))
            _, chunk_charges = modelc((chunk_species, chunk_coordinates))
            predicted_charges = torch.cat((predicted_charges, torch.flatten(chunk_charges)))
            chunk_num_atoms = (chunk_species >= 0).to(chunk_true_charges.dtype).sum()
            num_atoms += chunk_num_atoms
        # predicted_charges = torch.flatten(torch.tensor(predicted_charges))
        # true_charges = torch.flatten(torch.tensor(true_charges))
        count += predicted_charges.shape[0]
        total_mse += mse_sum(true_charges, predicted_charges).sum() / num_atoms

    return math.sqrt(total_mse)

###############################################################################
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# In the training loop, we need to compute force, and loss for forces
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 150
early_stopping_learning_rate = 1.0E-5
charge_coefficient = 1.0  # controls the importance of energy loss vs charge loss
best_model_checkpoint = 'charge-training-best.pt'

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
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    # Besides being stored in x, species and coordinates are also stored in y.
    # So here, for simplicity, we just ignore the x and use y for everything.
    for i, (_, batch_y) in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):

        true_energies = batch_y['energies']
        predicted_energies = []
        num_atoms = []
        charge_loss = []

        for chunk in batch_y['atomic']:
            chunk_species = chunk['species']
            chunk_coordinates = chunk['coordinates']
            chunk_true_charges = chunk['cm5']
            chunk_num_atoms = (chunk_species >= 0).to(true_energies.dtype).sum(dim=1)
            num_atoms.append(chunk_num_atoms)

            # We do not need grad for `chunk_coordinates`
            chunk_coordinates.requires_grad_(False)

            _, chunk_charges = modelc((chunk_species, chunk_coordinates))
            # _, chunk_energies = model((chunk_species, chunk_coordinates))

            # Now let's compute loss for force of this chunk
            chunk_charge_loss = mse(chunk_true_charges, chunk_charges).sum(dim=1) / chunk_num_atoms

            # predicted_energies.append(chunk_energies)
            charge_loss.append(chunk_charge_loss)

        num_atoms = torch.cat(num_atoms)
        # predicted_energies = torch.cat(predicted_energies)

        # Now the total loss has two parts, energy loss and charge loss
        # energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        charge_loss = torch.cat(charge_loss).mean()
        # loss = energy_loss + charge_coefficient * charge_loss
        loss = charge_coefficient * charge_loss

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
