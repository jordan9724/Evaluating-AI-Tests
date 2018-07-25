from tqdm import trange


class TestBase:
    epochs = None
    training_per_epoch = None
    test_info = None

    def get_actions(self):
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
        for epoch in trange(self.epochs):
            print('Epoch', epoch)
            for train_num in trange(self.training_per_epoch):
                yield epoch + 1, train_num
            print('\n')
            print('\n')

    def reset_after_terminal(self):
        pass

    def finish(self):
        pass

    @property
    def is_terminal(self):
        raise NotImplemented


# epochs = 5
# episodes = 10
# training_sets = 5
#
#
# def trainer():
#     for epoch in range(epochs):
#         def ep():
#             for episode in range(episodes):
#                 yield episode
#         yield epoch, 1, ep()
#         def tr():
#             for train in range(training_sets):
#                 yield train
#         yield epoch, 0, tr()
#
#
# for epoch, is_episode, get_data in trainer():
#     print('Epoch:', epoch)
#     if is_episode:
#         for data in get_data:
#             print('Episode:', data)
#     else:
#         for data in get_data:
#             print('Test Set:', data)

def gener():
    temp1 = []
    def _t(val):
        assert len(temp1) == 0, 'You can only do this once!'
        temp1.append(val)
    yield _t
    assert len(temp1) == 1, 'You must call the function!'
    print('Value', temp1[0])

