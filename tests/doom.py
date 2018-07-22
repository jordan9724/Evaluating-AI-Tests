import itertools as it

from tests.base import TestBase
from vizdoom.vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution


class DoomBase(TestBase):
    epochs = 20
    training_per_epoch = 2000
    frame_repeat = 12

    def __init__(self, scenario):
        self.config_file_path = "../scenarios/{}.cfg".format(scenario)

        print('Getting Actions...')
        self.game = DoomGame()
        self.game.load_config(self.config_file_path)
        self._actions = [list(act) for act in it.product([0, 1], repeat=self.game.get_available_buttons_size())]
        self.game.close()
        print('Done.')

    @property
    def is_terminal(self):
        return self.game.is_episode_finished()

    def get_actions(self):
        return self._actions

    def get_data(self):
        return self.game.get_state().screen_buffer

    def perform_action_and_get_reward(self, action):
        reward = self.game.make_action(self._actions[action], self.frame_repeat)
        return reward

    def run_results(self):
        if self.game.is_episode_finished():
            score = self.game.get_total_reward()

    def initialize_training(self):
        print('Initializing doom...')
        self.game = DoomGame()
        self.game.load_config(self.config_file_path)
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()
        print('Done.')


class DoomBasic(DoomBase):

    def __init__(self):
        super().__init__('basic')
