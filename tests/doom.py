import numpy as np
import itertools as it
import skimage.color
import skimage.transform

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
        self.actions = [list(act) for act in it.product([0, 1], repeat=self.game.get_available_buttons_size())]
        self.game.close()
        print('Done.')

    def train(self):

        for epoch in range(self.epochs):
            for train_num in range(self.training_per_epoch):
                pass

    # def preprocess(self, res):
    #     img = self.game.get_state().screen_buffer
    #     img = skimage.transform.resize(img, res)
    #     img = img.astype(np.float32)
    #     return img
    #
    # def get_reward(self, action):
    #     return self.game.make_action(self.actions[action], self.frame_repeat)
    #
    # @property
    # def is_terminal(self):
    #     return self.game.is_episode_finished()
    #
    # def initialize_training(self):
    #     print('Initializing doom...')
    #     game = DoomGame()
    #     game.load_config(self.config_file_path)
    #     game.set_window_visible(False)
    #     game.set_mode(Mode.PLAYER)
    #     game.set_screen_format(ScreenFormat.GRAY8)
    #     game.set_screen_resolution(ScreenResolution.RES_640X480)
    #     game.init()
    #     print('Done.')


class DoomBasic(DoomBase):

    def __init__(self):
        super().__init__('basic')
