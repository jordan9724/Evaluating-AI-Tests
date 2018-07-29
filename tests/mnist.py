from tensorflow.examples.tutorials.mnist import input_data

from tests.base import TestBase


class MNIST(TestBase):

    def __init__(self):
        self.mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
        self._img = None
        self._ans = None

    @staticmethod
    def to_str():
        return 'MNIST'

    def get_num_actions(self):
        return 10

    def train(self):
        self.initialize_training()
        for epoch in range(self.epochs):
            batch = self.mnist_data.train.next_batch(self.training_per_epoch)
            for train_num, (img, ans) in enumerate(zip(batch[0], batch[1])):
                self._img = img
                self._ans = ans
                yield epoch + 1, train_num

    def get_data(self):
        return self._img

    def perform_action_and_get_reward(self, action_idx):
        return self._ans[action_idx]

    @property
    def is_terminal(self):
        return True

    @staticmethod
    def get_min_max_score():
        return 0, 1
