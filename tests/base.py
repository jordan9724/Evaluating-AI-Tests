from tqdm import trange


EPOCHS = 20
TRAIN_PER_EPOCH = 2000


class TestBase:
    epochs = EPOCHS
    training_per_epoch = TRAIN_PER_EPOCH
    test_info = None

    @staticmethod
    def to_str():
        raise NotImplemented

    def get_num_actions(self):
        raise NotImplemented

    def get_data(self):
        raise NotImplemented

    def perform_action_and_get_reward(self, action_idx):
        raise NotImplemented

    def initialize_training(self):
        pass

    def set_test_info(self, test_info):
        self.test_info = test_info

    def after_epoch(self):
        pass

    def train(self):
        self.initialize_training()
        for epoch in range(self.epochs):
            for train_num in range(self.training_per_epoch):
                yield epoch + 1, train_num

    def reset_after_terminal(self):
        pass

    def finish(self):
        pass

    @property
    def is_terminal(self):
        raise NotImplemented

    @staticmethod
    def get_min_max_score():
        raise NotImplemented
