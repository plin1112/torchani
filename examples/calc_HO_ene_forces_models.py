import os
import numpy as np
import torch
import torchani
from torchani.data._pyanitools import anidataloader
from torchani.data._pyanitools import datapacker

code = {"1" : "H", "2" : "He", "3" : "Li", "4" : "Be", "5" : "B", \
"6"  : "C", "7"  : "N", "8"  : "O",  "9" : "F", "10" : "Ne", \
"11" : "Na" , "12" : "Mg" , "13" : "Al" , "14" : "Si" , "15" : "P", \
"16" : "S"  , "17" : "Cl" , "18" : "Ar" , "19" : "K"  , "20" : "Ca", \
"21" : "Sc" , "22" : "Ti" , "23" : "V"  , "24" : "Cr" , "25" : "Mn", \
"26" : "Fe" , "27" : "Co" , "28" : "Ni" , "29" : "Cu" , "30" : "Zn", \
"31" : "Ga" , "32" : "Ge" , "33" : "As" , "34" : "Se" , "35" : "Br", \
"36" : "Kr" , "37" : "Rb" , "38" : "Sr" , "39" : "Y"  , "40" : "Zr", \
"41" : "Nb" , "42" : "Mo" , "43" : "Tc" , "44" : "Ru" , "45" : "Rh", \
"46" : "Pd" , "47" : "Ag" , "48" : "Cd" , "49" : "In" , "50" : "Sn", \
"51" : "Sb" , "52" : "Te" , "53" : "I"  , "54" : "Xe" , "55" : "Cs", \
"56" : "Ba" , "57" : "La" , "58" : "Ce" , "59" : "Pr" , "60" : "Nd", \
"61" : "Pm" , "62" : "Sm" , "63" : "Eu" , "64" : "Gd" , "65" : "Tb", \
"66" : "Dy" , "67" : "Ho" , "68" : "Er" , "69" : "Tm" , "70" : "Yb", \
"71" : "Lu" , "72" : "Hf" , "73" : "Ta" , "74" : "W"  , "75" : "Re", \
"76" : "Os" , "77" : "Ir" , "78" : "Pt" , "79" : "Au" , "80" : "Hg", \
"81" : "Tl" , "82" : "Pb" , "83" : "Bi" , "84" : "Po" , "85" : "At", \
"86" : "Rn" , "87" : "Fr" , "88" : "Ra" , "89" : "Ac" , "90" : "Th", \
"91" : "Pa" , "92" : "U"  , "93" : "Np" , "94" : "Pu" , "95" : "Am", \
"96" : "Cm" , "97" : "Bk" , "98" : "Cf" , "99" : "Es" ,"100" : "Fm", \
"101": "Md" ,"102" : "No" ,"103" : "Lr" ,"104" : "Rf" ,"105" : "Db", \
"106": "Sg" ,"107" : "Bh" ,"108" : "Hs" ,"109" : "Mt" ,"110" : "Ds", \
"111": "Rg" ,"112" : "Uub","113" : "Uut","114" : "Uuq","115" : "Uup", \
"116": "Uuh","117" : "Uus","118" : "Uuo"}

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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

H_network = torch.nn.Sequential(
    torch.nn.Linear(128, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(128, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, O_network])

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torchani.nn.Sequential(aev_computer, nn, energy_shifter).to(device)

best_model_checkpoint = 'HO_small_forces-training-best.pt'

if os.path.isfile(best_model_checkpoint):
    checkpoint = torch.load(best_model_checkpoint, map_location=device)
    nn.load_state_dict(checkpoint)
    nn.eval()

count = 0
# log_file = 'ani_HO_ene_forces_wb97x_dz_set.log'
log_file = 'ani2x_HO_ene_forces_wb97x_dz_set.log'
f_log = open(log_file, 'w+')

dpack1 = datapacker('./ani2x_HO_ene_forces_wb97x_dz_set.h5')

for f_h5 in ['../dataset/ani2x-20191001_HO_wb97x_dz_force.h5']: 
    for data in anidataloader(f_h5):
        formula = data['path'].split('/')[-1]
        species = data['species']
        coordinates = data['coordinates']
        wb97x_dz_energy = data['energies']
        wb97x_dz_forces = data['forces']

        species_list = [x for x in species] # [code.get(str(x), 'X') for x in species]
        species_tensor = species_to_tensor(species_list).to(device).unsqueeze(0)

        if len(coordinates) > 0:
            species_tensor = species_tensor.expand(len(coordinates), -1)
            coords = torch.tensor(coordinates, requires_grad=True, device=device)
            f_log.write('##### Calculate model energies and forces on %s ##### \n' % formula)
            count += 1
            model_energies = model((species_tensor, coords)).energies
            model_forces = -torch.autograd.grad(model_energies.sum(), coords, create_graph=True, retain_graph=True)[0]

        model0_energies = model_energies.detach().cpu().numpy()
        model0_forces = model_forces.detach().cpu().numpy()

        del species_tensor
        del coords
        del model_energies
        del model_forces
        torch.cuda.empty_cache()

        dpack1.store_data(data['path'],
                          species = species,
                          coordinates = coordinates,
                          wb97x_dz_energy = wb97x_dz_energy,
                          model_energy = model0_energies,
                          wb97x_dz_forces = wb97x_dz_forces,
                          model_forces = model0_forces,
                         )

print('total %d calculations completed' % count)

dpack1.cleanup()
f_log.close()

