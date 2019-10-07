# -*- coding: utf-8 -*-
"""
Use Disk Cache of AEV to Boost Training
=======================================

In the previous :ref:`training-example` example, AEVs are computed everytime
when needed. This is not very efficient because the AEVs actually never change
during training. If one has a good SSD, it would be beneficial to cache these
AEVs.  This example shows how to use disk cache to boost training
"""

###############################################################################
# Most part of the codes in this example are line by line copy of
# :ref:`training-example`.
import torch
import torchani
import os


# training and validation set
try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
training_path = os.path.join(path, '../dataset/ani-1x/ani_al-901_training.h5')
validation_path = os.path.join(path, '../dataset/ani-1x/ani_al-901_validation.h5')  # noqa: E501

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch size
batch_size = 256

###############################################################################
# Here, there is no need to manually construct aev computer and energy shifter,
# but we do need to generate a disk cache for datasets
const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')
sae_file = os.path.join(path, '../torchani/resources/ani-1x_8x/sae_linfit.dat')
training_cache = './ani_al-901_training_cache'
validation_cache = './ani_al-901_validation_cache'

# If the cache dirs already exists, then we assume these data has already been
# cached and skip the generation part.
if not os.path.exists(training_cache):
    torchani.data.cache_aev(training_cache, training_path, batch_size, device,
                            const_file, True, sae_file,
                            properties=('energies', 'dipole'), atomic_properties=('cm5', 'hirshfeld', 'forces')
                            )
if not os.path.exists(validation_cache):
    torchani.data.cache_aev(validation_cache, validation_path, batch_size,
                            device, const_file, True, sae_file,
                            properties=('energies', 'dipole'), atomic_properties=('cm5', 'hirshfeld', 'forces')
                            )
