#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from random import sample, randint, random
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf

from models.base import ModelBase
from tools.save_file import SaveInfo


class ReplayMemory:
    def __init__(self, capacity, resolution):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class CNN(ModelBase):

    def __init__(self, load_from_num=None):
        # Q-learning settings
        self.learning_rate = 0.00025
        self.discount_factor = 0.99
        self.epochs = 20
        self.learning_steps_per_epoch = 2000
        self.replay_memory_size = 10000

        # NN learning settings
        self.batch_size = 64

        # Training regime
        self.test_episodes_per_epoch = 100

        # Other parameters
        self.frame_repeat = 12
        self.resolution = (30, 45)
        self.episodes_to_watch = 10

        # Net setup
        self.memory = ReplayMemory(self.replay_memory_size, self.resolution)
        self.learn = None
        self.get_q_values = None
        self.get_best_action = None
        self.s1 = None

        # Other settings
        self.session = None
        self.save_info = SaveInfo(save_type='weights', save_num=load_from_num, extra_info='cnn.dump')
        self.is_loading = load_from_num is not None
        self.saver = None

    def set_test_info(self, test_info):
        super().set_test_info(test_info)
        self.setup_net()

    def setup_net(self):
        self.session = tf.Session()
        self.learn, self.get_q_values, self.get_best_action = self.create_network()
        self.saver = tf.train.Saver()

        # Restore weights if applicable
        if self.is_loading:
            self.load_model()
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)

    def get_action(self):
        # Define exploration rate change over time
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * self.epochs  # 60% of learning time

        if self.test_info.curr_epoch < const_eps_epochs:
            eps = start_eps
        elif self.test_info.curr_epoch < eps_decay_epochs:
            # Linear decay
            eps = start_eps - (self.test_info.curr_epoch - const_eps_epochs) / \
                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            eps = end_eps

        self.s1 = self.preprocess(self.test_info.data)
        if random() <= eps:
            return randint(0, self.test_info.num_actions - 1)
        else:
            return self.get_best_action(self.s1)

    def receive_reward(self):
        s2 = self.preprocess(self.test_info.data) if not self.test_info.is_terminal else None

        self.memory.add_transition(
            self.s1,
            self.test_info.last_action_idx,
            s2,
            self.test_info.is_terminal,
            self.test_info.last_reward
        )
        self.learn_from_memory()

    def save_model(self):
        self.saver.save(self.session, self.save_info.get_file_name())
        pass

    def load_model(self):
        self.saver.restore(self.session, self.save_info.get_file_name())

    def preprocess(self, data):
        # Converts and down-samples the input image
        data = skimage.transform.resize(data, self.resolution)
        data = data.astype(np.float32)
        return data

    def create_network(self):
        # Set num_outputs based off of handicap - num_outputs in [1, 2, 4, 8, 16, 32, 64, 128]
        num_outputs = 2 ** int((1 - self.handicap) * 8)

        # Create the input variables
        s1_ = tf.placeholder(tf.float32, [None] + list(self.resolution) + [1], name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, self.test_info.num_actions], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=num_outputs, kernel_size=[6, 6], stride=[3, 3],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=num_outputs, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.test_info.num_actions, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
        best_a = tf.argmax(q, 1)

        loss = tf.losses.mean_squared_error(q, target_q_)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def function_learn(s1, target_q):
            feed_dict = {s1_: s1, target_q_: target_q}
            l, _ = self.session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state):
            return self.session.run(q, feed_dict={s1_: state})

        def function_get_best_action(state):
            return self.session.run(best_a, feed_dict={s1_: state})

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, self.resolution[0], self.resolution[1], 1]))[0]

        return function_learn, function_get_q_values, function_simple_get_best_action

    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

            q2 = np.max(self.get_q_values(s2), axis=1)
            target_q = self.get_q_values(s1)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
            self.learn(s1, target_q)
