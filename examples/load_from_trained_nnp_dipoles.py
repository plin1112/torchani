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
from torchani.data._pyanitools import anidataloader
from torchani.data._pyanitools import datapacker

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

    def dipole_moment(self, coords_charges):
        """
        Calculates dipole moment of the molecule.
        coords_charges: tuple of atomic coordinates and point charges
        coords is a 3-dimension tensor (mol, atom, xyz)
        charges is a 2-dimension tensor (mol, atom)
        """
        # toDebye = 4.80320425 : use if the labels are in the unit of Debye
        coords, charges = coords_charges
        charges = charges.unsqueeze(-1)
        dipoles = coords * charges
        dipoles = torch.sum(dipoles, dim=1)
        return dipoles

    def forward(self, species_aev_coordinates):
        species, aev, coordinates = species_aev_coordinates
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
        return species, output, self.dipole_moment((coordinates, output)), excess_charge.squeeze(-1)

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
    aevp_computer = torchani.aev.AEVPComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    energy_shifter = torchani.utils.EnergyShifter(None)
    species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
# dspath = os.path.join(path, '../dataset/ani-1x/ani_al-901_validation.h5')
dspath = os.path.join(path, '../dataset/COMP6/COMP6v1/DrugBank/drugbank_testset_mod.h5')

# checkpoint file for best model and latest model
best_model_checkpoint = 'dipole-transfer-training-best0-4.pt'
# latest_checkpoint = 'charge-transfer-training-latest.pt'

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

# nn = ANICModel([H_network, C_network, N_network, O_network])
nn = ANIDModel([H_network, C_network, N_network, O_network])

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
# model = torch.nn.Sequential(aev_computer, nn).to(device)
modelc = torch.nn.Sequential(aevp_computer, nn).to(device)

###############################################################################
# This part of the code is also the same

if os.path.isfile(best_model_checkpoint):
    checkpoint = torch.load(best_model_checkpoint)
    nn.load_state_dict(checkpoint)

# dpack1 = datapacker('./ani_al-901_validation_charge_dipole_trained_nnp_dipoles_on_ani_al_901_v4.h5')
dpack1 = datapacker('./drugbank_testset_mod_nnp_dipoles_on_ani_al_901_v4.h5')

for data in anidataloader(dspath):
    formula = data['path'].split('/')[-1]
    species_list = data['species']
    coordinates = data['coordinates']
    cm5 = data['cm5']
    # dipole = data['dipole']
    energies = data['energies']
    hirshfeld = data['hirshfeld']

    species_tensor = species_to_tensor(species_list).to(device).unsqueeze(0)
    species_tensor = species_tensor.expand(len(coordinates), -1)
    coords = torch.tensor(coordinates, requires_grad=False, device=device)
    _, modelc_charges, modelc_dipoles, modelc_total_charges = modelc((species_tensor, coords))

    new_charges = modelc_charges.detach().cpu().numpy()
    new_dipoles = modelc_dipoles.detach().cpu().numpy()
    new_total_charges = modelc_total_charges.detach().cpu().numpy()

    dpack1.store_data(data['path'],
                      species = species_list,
                      coordinates = coordinates,
                      wb97x_dz_energy = energies,
                      cm5 = cm5,
                      # dipole = dipole,
                      hirshfeld = hirshfeld,
                      nnp_charges = new_charges,
                      nnp_dipoles = new_dipoles,
                      nnp_total_charges = new_total_charges,
                      )

dpack1.cleanup()


# f0_out.close()
# f1_out.close()