

class TestBase:
    epochs = None
    training_per_epoch = None

    def train(self):
        """
        Yields (tools, get_reward_func: get_reward, epoch, max_epoch)
        """
        raise NotImplemented

    def get_reward(self, action):
        """
        Returns (reward, is_terminal)

        TODO: Might put in `train`
        """
        raise NotImplemented

    def is_terminal(self):
        raise NotImplemented

    def get_actions(self):
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
#         for tools in get_data:
#             print('Episode:', tools)
#     else:
#         for tools in get_data:
#             print('Test Set:', tools)

def gener():
    temp1 = []
    def _t(val):
        assert len(temp1) == 0, 'You can only do this once!'
        temp1.append(val)
    yield _t
    assert len(temp1) == 1, 'You must call the function!'
    print('Value', temp1[0])

