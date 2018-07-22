from models.base import ModelBase
from tests.doom import DoomBasic

TESTS = (
    DoomBasic,
)


def run_default():
    for test in TESTS:
        yield test()


class TestRunner:
    curr_test = None
    model = None

    def __init__(self, model: ModelBase):
        self.model = model

    def start(self):
        self.run_test(run_default())

    def run_test(self, tests: iter):
        # Runs through each tests
        for test in tests:
            # Sets the models actions
            self.model.set_actions(test.get_actions())

            # Loops through each training set
            for epoch, train_num in test.train():

                # Model chooses action based off data
                model_action = self.model.get_action(True, test.get_data(), epoch, test.epochs)

                # Model gets reward from chosen action
                reward = test.perform_action_and_get_reward(model_action)
                self.model.receive_reward(True, test.get_data(), reward, test.is_terminal)
