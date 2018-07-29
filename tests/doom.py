import itertools as it

from tests.base import TestBase
from vizdoom.vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# from tools.analysis import DataAnalyzer
from tools.save_file import SaveInfo


class DoomBase(TestBase):
    frame_repeat = 12

    def __init__(self, scenario, graph_sub_title=None):
        save_info = SaveInfo('scenarios', save_num=False, extra_info='{}.cfg'.format(scenario))
        self.config_file_path = save_info.get_file_name()
        # self.analyzer = DataAnalyzer(
        #     self.epochs * self.training_per_epoch,
        #     ["Epoch", "Train", "Score", "Finished"],
        #     extra_info='{}_data'.format(scenario),
        #     graph_sub_title=graph_sub_title
        # )

        print('Getting Actions...')
        self.game = DoomGame()
        self.game.load_config(self.config_file_path)
        self._actions = [list(act) for act in it.product([0, 1], repeat=self.game.get_available_buttons_size())]
        self.game.close()
        print('Done.')

    @property
    def is_terminal(self):
        return self.game.is_episode_finished()

    def get_num_actions(self):
        return len(self._actions)

    def get_data(self):
        return self.game.get_state().screen_buffer.copy()

    def perform_action_and_get_reward(self, action_idx):
        reward = self.game.make_action(self._actions[action_idx], self.frame_repeat)
        # self.analyzer.set_next_values([self.test_info.curr_epoch, self.test_info.curr_train, reward, self.test_info.is_terminal])
        return reward

    # def after_epoch(self):
    #     print(np.average(self.scores[self.test_info.curr_epoch]))

    # def finish(self):
    #     self.analyzer.display_specific_epoch_vs_score()
        # self.analyzer.display_error_bar_over_many("Epoch", "Score", "Epoch vs Score", x_skip=1, y_skip=25)
        # self.analyzer.display_sum_y_vals("Epoch", "Finished", "Epoch vs Finished", x_skip=1, y_skip=10)

    def reset_after_terminal(self):
        self.game.new_episode()

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

    def __init__(self, graph_sub_title=None):
        super().__init__('basic', graph_sub_title)

    @staticmethod
    def to_str():
        return 'Doom Basic'

    @staticmethod
    def get_min_max_score():
        return -400, 100
        # self.analyzer.display_specific_epoch_vs_score(min_score, max_score)
        # self.analyzer.display_specific_epoch_vs_std(min_score, max_score)


class DoomPredictPosition(DoomBase):

    def __init__(self, graph_sub_title=None):
        super().__init__('predict_position', graph_sub_title)

    @staticmethod
    def to_str():
        return 'Doom Predict Position'

    @staticmethod
    def get_min_max_score():
        return -0.3, 1
        # self.analyzer.display_specific_epoch_vs_score(min_score, max_score)
        # self.analyzer.display_specific_epoch_vs_std(min_score, max_score)
