from models.base import ModelBase
from tests.doom import DoomBasic

TESTS = (
    DoomBasic,
)


def run_default():
    for test in TESTS:
        yield test


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
            for data, get_reward_func, epoch, max_epoch in test.train():

                # Model chooses action based off tools
                model_action = self.model.get_action(True, data, epoch, max_epoch)

                # Model gets reward from chosen action
                reward, is_terminal = get_reward_func(model_action)
                self.model.give_reward(reward, is_terminal)
