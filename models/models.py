import pickle
import skimage.transform
import numpy as np
# import theano
from random import random, randint

# from lasagne.init import HeUniform, Constant
# from lasagne.layers import get_all_param_values, set_all_param_values, InputLayer, Conv2DLayer, DenseLayer, get_output, \
#     get_all_params
# from lasagne.nonlinearities import rectify
# from lasagne.objectives import squared_error
# from lasagne.updates import rmsprop

from models.base import ModelBase
# from models._cnn import ReplayMemory
# from theano import tensor

from tools.save_file import SaveInfo


class FirstAction(ModelBase):

    def get_action(self, **kwargs):
        return self.actions[0]


class RandAction(ModelBase):

    def get_action(self, **kwargs):
        return self.actions[random.randint(0, len(self.actions) - 1)]
