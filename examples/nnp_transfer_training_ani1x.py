# -*- coding: utf-8 -*-
"""
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

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()

const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
consts = torchani.neurochem.Constants(const_file)

# save existing model parameters into checkpoint file format
for i in range(8):
    model_dir = os.path.join(path, '../torchani/resources/ani-1x_8x/train'+str(i)+'/networks')  # noqa: E501
    model = torchani.neurochem.load_model(consts.species, model_dir)

    ani_1x_model = 'ani-1x_8x_model'+str(i)+'.pt'
    torch.save(model.state_dict(), ani_1x_model)
