import pickle
import skimage.transform
import numpy as np
import theano

from random import random, randint

from lasagne.init import HeUniform, Constant
from lasagne.layers import get_all_param_values, set_all_param_values, InputLayer, Conv2DLayer, DenseLayer, get_output, \
    get_all_params
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop

from models.base import ModelBase
from models.cnn_dependents import ReplayMemory
from theano import tensor

from tests.base import TestBase
from tools.save_file import SaveInfo


class FirstAction(ModelBase):

    def get_action(self, **kwargs):
        return self.actions[0]


class RandAction(ModelBase):

    def get_action(self, **kwargs):
        return self.actions[random.randint(0, len(self.actions) - 1)]


class CNN(ModelBase):
    batch_size = 64
    learning_rate = 0.00025
    discount_factor = 0.99
    replay_memory_size = 10000
    resolution = (30, 45)
    epochs = 5
    data_before_action = None
    curr_action = None

    def __init__(self, test_instance: TestBase, num_filters=8):
        print('Initializing the CNN...')
        super().__init__(test_instance)
        self.num_filters = num_filters
        self._save_info = SaveInfo('weights', extra_info='cnn_f{}.dump'.format(self.num_filters))

        # Create replay memory which will store the transitions
        self.memory = ReplayMemory(capacity=self.replay_memory_size, resolution=self.resolution)

        self.net, self.learn, self.get_q_values, self.get_best_action = self.create_network(len(self.actions))
        print('Done.')

    def get_action(self, is_training: bool, data, epoch=None, max_epoch=None):
        self.data_before_action = self.preprocess(data)
        if is_training:
            exploration_rate = self.get_exploration_rate(epoch, max_epoch)

            if random() <= exploration_rate:
                self.curr_action = randint(0, len(self.actions) - 1)
                return self.curr_action

        self.curr_action = self.get_best_action(self.data_before_action)
        return self.curr_action

    def receive_reward(self, is_training: bool, data, reward: float, is_terminal: bool):
        if is_training:
            data_after_action = None
            if not is_terminal and data is not None:
                data_after_action = self.preprocess(data)

            self.memory.add_transition(self.data_before_action, self.curr_action, data_after_action, is_terminal, reward)
            self.learn_from_memory()

    @staticmethod
    def get_exploration_rate(epoch=None, max_epoch=None):
        exploration_rate = 0.85
        if epoch is not None and max_epoch is not None:
            # Changes exploration rate over time
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * max_epoch  # 10% of learning time
            eps_decay_epochs = 0.6 * max_epoch  # 60% of learning time

            if epoch < const_eps_epochs:
                exploration_rate = start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                exploration_rate = start_eps - (epoch - const_eps_epochs) / \
                       (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                exploration_rate = end_eps

        return exploration_rate

    def preprocess(self, data):
        data = skimage.transform.resize(data, self.resolution)
        data = data.astype(np.float32)
        return data

    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

            q2 = np.max(self.get_q_values(s2), axis=1)
            # the value of q2 is ignored in learn if s2 is terminal
            self.learn(s1, q2, a, r, isterminal)

    def save_net(self):
        print("Saving the network weights to:", self._save_info.get_file_name())
        pickle.dump(get_all_param_values(self.net), open(self._save_info.get_file_name(), "wb"))

    def load_net(self):
        if not self._save_info.get_file_name():
            raise Exception("No save file specified")

        print("Loading the network weights from:", self._save_info.get_file_name())
        params = pickle.load(open(self._save_info.get_file_name(), "rb"))
        set_all_param_values(self.net, params)

    def create_network(self, available_actions_count):
        # Create the input variables
        s1 = tensor.tensor4("State")
        a = tensor.vector("Action", dtype="int32")
        q2 = tensor.vector("Q2")
        r = tensor.vector("Reward")
        isterminal = tensor.vector("IsTerminal", dtype="int8")

        # Create the input layer of the network.
        dqn = InputLayer(shape=[None, 1, self.resolution[0], self.resolution[1]], input_var=s1)

        # Add 2 convolutional layers with ReLu activation
        dqn = Conv2DLayer(dqn, num_filters=self.num_filters, filter_size=[6, 6],
                          nonlinearity=rectify, W=HeUniform("relu"),
                          b=Constant(.1), stride=3)
        dqn = Conv2DLayer(dqn, num_filters=self.num_filters, filter_size=[3, 3],
                          nonlinearity=rectify, W=HeUniform("relu"),
                          b=Constant(.1), stride=2)

        # Add a single fully-connected layer.
        dqn = DenseLayer(dqn, num_units=128, nonlinearity=rectify, W=HeUniform("relu"),
                         b=Constant(.1))

        # Add the output layer (also fully-connected).
        # (no nonlinearity as it is for approximating an arbitrary real function)
        dqn = DenseLayer(dqn, num_units=available_actions_count, nonlinearity=None)

        # Define the loss function
        q = get_output(dqn)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + self.discount_factor * (1 - isterminal) * q2)
        loss = squared_error(q, target_q).mean()

        # Update the parameters according to the computed gradient using RMSProp.
        params = get_all_params(dqn, trainable=True)
        updates = rmsprop(loss, params, self.learning_rate)

        # Compile the theano functions
        print("Compiling the network ...")
        function_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
        function_get_q_values = theano.function([s1], q, name="eval_fn")
        function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
        print("Network compiled.")

        def simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, 1, self.resolution[0], self.resolution[1]]))

        # Returns Theano objects for the net and functions.
        return dqn, function_learn, function_get_q_values, simple_get_best_action


