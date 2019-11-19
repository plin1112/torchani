# -*- coding: utf-8 -*-
"""
.. _force-training-example:

Train Neural Network Potential To Both Energies and Forces
==========================================================

We have seen how to train a neural network potential by manually writing
training loop in :ref:`training-example`. This tutorial shows how to modify
that script to train to force.
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
num_species = 2
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter([-0.600953, -75.194466])
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HO')

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath = os.path.join(path, '../dataset/ani1x-20191001_HO_wb97x_dz_forces_small.h5')

batch_size = 2560
chunk_threshold = 5

'''
other_properties = {'properties': ['dipoles', 'forces', 'energies'],
                    'padding_values': [None, 0, None],
                    'padded_shapes': [(batch_size, 3), (batch_size, -1, 3), (batch_size, )],
                    'dtypes': [torch.float32, torch.float32, torch.float64],
                    }
'''

# '''
other_properties = {'properties': ['forces', 'energies'],
                    'padding_values': [0, None],
                    'padded_shapes': [(batch_size, -1, 3), (batch_size, )],
                    'dtypes': [torch.float32, torch.float64],
                    }

# '''

'''
 other_properties = {'properties': ['energies'],
                    'padding_values': [None],
                    'padded_shapes': [(batch_size, )],
                    'dtypes': [torch.float64],
                    }
'''

ds = torchani.data.CachedDataset(dspath, batch_size=batch_size, device=device,
                                 chunk_threshold=chunk_threshold,
                                 other_properties=other_properties,
                                 species_order=['H', 'O'],
                                 self_energies=[-0.600953, -75.194466],
                                 subtract_self_energies=True)
# chunks, properties = ds[0]
ds.load()
train_dataset, val_dataset = ds.split(0.2)
# '''

###############################################################################
# The code to create the dataset is a bit different: we need to manually
# specify that ``atomic_properties=['forces']`` so that forces will be read
# from hdf5 files.

'''
train_dataset, val_dataset  = torchani.data.load_ani_dataset(
    dspath, species_to_tensor, batch_size, rm_outlier=True,
    device=device, atomic_properties=['forces'],
    transform=[energy_shifter.subtract_from_dataset], split=[0.8, None])
'''

print('Self atomic energies: ', energy_shifter.self_energies)

###############################################################################
# The code to define networks, optimizers, are mostly the same
dropout_prob = 0.2

H_network = torch.nn.Sequential(
    torch.nn.Linear(128, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Dropout(p=dropout_prob),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Dropout(p=dropout_prob),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(128, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Dropout(p=dropout_prob),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Dropout(p=dropout_prob),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, O_network])
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
model = torchani.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Here we will use Adam with weight decay for both the weights and biases.

AdamW = torchani.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[2].bias], 'weight_decay': 0.00001},
    {'params': [H_network[5].weight], 'weight_decay': 0.000001},
    {'params': [H_network[5].bias], 'weight_decay': 0.000001},
    {'params': [H_network[8].weight]},
    {'params': [H_network[8].bias]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[2].bias], 'weight_decay': 0.00001},
    {'params': [O_network[5].weight], 'weight_decay': 0.000001},
    {'params': [O_network[5].bias], 'weight_decay': 0.000001},
    {'params': [O_network[8].weight]},
    {'params': [O_network[8].bias]},
])

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)

###############################################################################
# This part of the code is also the same
latest_checkpoint = 'HO_small_do_newds_forces-training-latest.pt'
best_model_checkpoint = 'HO_small_do_newds_forces-training-best.pt'

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    nn.eval()

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
    for batch_x, batch_y in val_dataset:
        true_energies = batch_y['energies'].float()
        predicted_energies = []
        for chunk_species, chunk_coordinates in batch_x:
            chunk_energies = model((chunk_species, chunk_coordinates)).energies
            predicted_energies.append(chunk_energies)
        predicted_energies = torch.cat(predicted_energies)
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcal(math.sqrt(total_mse / count))


###############################################################################
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# In the training loop, we need to compute force, and loss for forces
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
# We only train 3 epoches here in able to generate the docs quickly.
# Real training should take much more than 3 epoches.
max_epochs = 8000
early_stopping_learning_rate = 1.0E-6
force_coefficient = 0.1  # controls the importance of energy loss vs force loss

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

    # Besides being stored in x, species and coordinates are also stored in y.
    # So here, for simplicity, we just ignore the x and use y for everything.
    for i, (batch_x, batch_y) in tqdm.tqdm(
        enumerate(train_dataset),
        total=len(train_dataset),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):

        true_energies = batch_y['energies'].float()
        true_forces = batch_y['forces']
        max_atoms = true_forces.shape[1]
        predicted_energies = []
        predicted_forces = []
        num_atoms = []
        ## force_loss = []

        for chunk_species, chunk_coordinates in batch_x:
            ## chunk_true_forces = chunk['forces']
            chunk_num_atoms = (chunk_species >= 0).to(true_energies.dtype).sum(dim=1)
            num_atoms.append(chunk_num_atoms)

            # We must set `chunk_coordinates` to make it requires grad, so
            # that we could compute force from it
            chunk_coordinates.requires_grad_(True)

            chunk_energies = model((chunk_species, chunk_coordinates)).energies
            predicted_energies.append(chunk_energies)

            # We can use torch.autograd.grad to compute force. Remember to
            # create graph so that the loss of the force can contribute to
            # the gradient of parameters, and also to retain graph so that
            # we can backward through it a second time when computing gradient
            # w.r.t. parameters.
            chunk_forces = -torch.autograd.grad(chunk_energies.sum(), chunk_coordinates, create_graph=True, retain_graph=True)[0]

            chunk_forces_shape = list(chunk_forces.shape)
            chunk_forces_shape[1] = max_atoms - chunk_forces_shape[1]
            padding = chunk_forces.new_full(chunk_forces_shape, 0)
            chunk_forces = torch.cat([chunk_forces, padding], dim=1)

            predicted_forces.append(chunk_forces)
            ## force_loss.append(chunk_force_loss)

        num_atoms = torch.cat(num_atoms)
        predicted_energies = torch.cat(predicted_energies)
        predicted_forces = torch.cat(predicted_forces, dim=0)
        # Now the total loss has two parts, energy loss and force loss
        energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        force_loss = (mse(true_forces, predicted_forces).sum(dim=(1, 2)) / num_atoms.sqrt()).mean()
        ## force_loss = torch.cat(force_loss).mean()
        loss = energy_loss + force_coefficient * force_loss

        AdamW.zero_grad()
        loss.backward()
        AdamW.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(train_dataset) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
    }, latest_checkpoint)


