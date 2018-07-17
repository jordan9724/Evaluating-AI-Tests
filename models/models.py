import os
import pickle
import random
import numpy as np
import theano

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


class FirstAction(ModelBase):

    def get_action(self, data):
        return self.test_instance.actions[0]


class RandAction(ModelBase):

    def get_action(self, data):
        return self.test_instance.actions[random.randint(0, len(self.test_instance.actions) - 1)]


class CNN(ModelBase):
    batch_size = 64
    learning_rate = 0.00025
    discount_factor = 0.99
    replay_memory_size = 10000
    resolution = (30, 45)
    epochs = 5

    @property
    def model_savefile(self):
        run_nums = [int(name.split('_')[0]) for name in os.listdir("../weights")]
        if len(run_nums) == 0:
            _curr_run = 0
        else:
            _curr_run = max(run_nums) + 1
        curr_run = "{:0>4s}".format(str(_curr_run))

        return "../nets/{}_cnn_f{}_e{}.dump".format(
            curr_run,
            self.num_filters,
            self.epochs
        )

    def __init__(self, test_instance: TestBase, num_filters=8):
        print('Initializing the CNN...')
        super().__init__(test_instance)
        self.num_filters = num_filters

        # Create replay memory which will store the transitions
        self.memory = ReplayMemory(capacity=self.replay_memory_size, resolution=self.resolution)

        self.net, self.learn, self.get_q_values, self.get_best_action = self.create_network(len(self.test_instance.actions))
        print('Done.')

    def save_net(self):
        print("Saving the network weights to:", self.model_savefile)
        pickle.dump(get_all_param_values(self.net), open(self.model_savefile, "wb"))

    def load_net(self):
        if not self.model_savefile:
            raise Exception("No save file specified")

        print("Loading the network weights from:", self.model_savefile)
        params = pickle.load(open(self.model_savefile, "rb"))
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

    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

            q2 = np.max(self.get_q_values(s2), axis=1)
            # the value of q2 is ignored in learn if s2 is terminal
            self.learn(s1, q2, a, r, isterminal)

    def perform_learning_step(self, epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(_epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * self.epochs  # 60% of learning time

            if _epoch < const_eps_epochs:
                return start_eps
            elif _epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (_epoch - const_eps_epochs) / \
                                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps

        s1 = self.test_instance.preprocess(self.resolution)

        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if random.random() <= eps:
            a = random.randint(0, len(self.test_instance.actions) - 1)
        else:
            # Choose the best action according to the network.
            a = self.get_best_action(s1)
        reward = self.test_instance.get_reward(a)

        s2 = self.test_instance.preprocess(self.resolution) if not self.test_instance.isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, self.test_instance.isterminal, reward)

        self.learn_from_memory()

    def train(self, learning_steps_per_epoch, test_episodes_per_epoch):
        print('{:=^40s}'.format(' Starting Training! '))

        self.test_instance.initialize_training()

        for epoch in range(self.epochs):
            print('{:=^40s}'.format(' Starting Epoch {} '.format(epoch)))


