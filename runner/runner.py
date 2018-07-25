from tests.doom import *

TESTS = (
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
    model = None

    def __init__(self, model):
        self.model = model

    def start(self):
        self.run_test(run_default())

    def run_test(self, tests: iter):

        # Runs through each tests
        for test in tests:
            test_info = TestInfo()

            # Sets the models actions
            test_info.num_actions = len(test.get_actions())
            test_info.max_epochs = test.epochs
            test_info.max_trains_per_epoch = test.training_per_epoch
            test_info.is_training = True
            test_info.total_reward = 0
            self.model.set_test_info(test_info)
            test.set_test_info(test_info)

            # Loops through each training set
            for epoch, train_num in test.train():
                # Saves the model on each epoch
                if test_info.curr_epoch != epoch:
                    test.after_epoch()
                    # self.model.save_model()
                    test_info.curr_epoch = epoch

                test_info.curr_train = train_num
                test_info.data = test.get_data()

                # Model chooses action based off data
                test_info.last_action_idx = self.model.get_action()

                # Model gets reward from chosen action
                test_info.last_reward = test.perform_action_and_get_reward(test_info.last_action_idx)
                test_info.total_reward += test_info.last_reward
                test_info.is_terminal = test.is_terminal
                if not test_info.is_terminal:
                    test_info.data = test.get_data()

                self.model.receive_reward()

                if test_info.is_terminal:
                    test.reset_after_terminal()
                    test_info.total_reward = 0

            test.finish()
