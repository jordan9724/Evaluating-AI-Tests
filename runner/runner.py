import numpy as np

from tests.doom import *
from tests.mnist import *

from tools.analysis import DataAnalyzer

num_handicaps = 8
HANDICAPS = [(num_handicaps - i) / num_handicaps for i in range(num_handicaps + 1)]
# HANDICAPS = [0]
TESTS = (
    MNIST,
    DoomBasic,
)


def run_default():
    for test in TESTS:
        yield test()


class TestInfo:

    def __init__(self):
        self.num_actions = 0
        self.max_epochs = 0
        self.max_trains_per_epoch = 0
        self.curr_epoch = 0
        self.curr_train = 0
        self.last_reward = 0
        self.total_reward = 0
        self.last_action_idx = 0
        self.is_training = False
        self.is_terminal = False
        self.data = None


class TestRunner:
    curr_test = None

    def __init__(self, model):
        self.model = model
        from tests.base import EPOCHS
        self.scores = {test.to_str(): {handicap: {epoch + 1: [] for epoch in range(EPOCHS)} for handicap in HANDICAPS} for test in TESTS}

    def run_tests(self):
        # Runs through each tests
        for _t in TESTS:
            test = _t()
            for handicap in HANDICAPS:
                model_instance = self.model()
                model_instance.set_handicap(handicap)
                test_info = TestInfo()

                # Sets the models actions
                test_info.num_actions = test.get_num_actions()
                test_info.max_epochs = test.epochs
                test_info.max_trains_per_epoch = test.training_per_epoch
                test_info.is_training = True
                test_info.total_reward = 0
                model_instance.set_test_info(test_info)
                test.set_test_info(test_info)

                # Loops through each training set
                for epoch, train_num in test.train():
                    # Saves the model on each epoch
                    if test_info.curr_epoch != epoch:
                        print('Handicap', handicap)
                        test.after_epoch()
                        model_instance.save_model()
                        test_info.curr_epoch = epoch

                    test_info.curr_train = train_num
                    test_info.data = test.get_data()

                    # Model chooses action based off data
                    test_info.last_action_idx = model_instance.get_action()

                    # Model gets reward from chosen action
                    test_info.last_reward = test.perform_action_and_get_reward(test_info.last_action_idx)
                    test_info.total_reward += test_info.last_reward
                    test_info.is_terminal = test.is_terminal
                    if not test_info.is_terminal:
                        test_info.data = test.get_data()

                    model_instance.receive_reward()

                    if test_info.is_terminal:
                        test.reset_after_terminal()
                        self.scores[test.to_str()][handicap][epoch].append(test_info.total_reward)
                        test_info.total_reward = 0

                test.finish()

        from tests.base import EPOCHS, TRAIN_PER_EPOCH
        da = DataAnalyzer(EPOCHS * len(HANDICAPS) * len(TESTS), ['Test', 'Min', 'Max', 'Epoch', 'Handicap', 'Score', 'STD', 'Finished'], extra_info=self.model.to_str(), graph_sub_title=self.model.to_str())
        for epoch in range(EPOCHS):
            for test in TESTS:
                for handicap in HANDICAPS:
                    scores = np.array(self.scores[test.to_str()][handicap][epoch + 1])
                    min_score, max_score = test.get_min_max_score()
                    da.set_next_values([test.to_str(), min_score, max_score, epoch, handicap, scores.mean(), scores.std(), len(scores)])

        # Assumes there was only one test - being lazy
        # min_score, max_score = test.get_min_max_score()
        da.display_epoch_handicap(TRAIN_PER_EPOCH)
