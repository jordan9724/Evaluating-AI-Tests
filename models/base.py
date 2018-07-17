class ModelBase:
    actions = None

    def get_action(self, is_training, data, epoch=None, max_epoch=None):
        """
        Gives back an action with assistance of the given tools

        :param is_training: True when the system should be learning
        :param data: Used to determine an action
        :param epoch: Current epoch
        :param max_epoch: Max number of epochs
        :return: The chosen action
        """
        raise NotImplemented

    def give_reward(self, reward, is_terminal):
        """
        After the action is received, the model will be given a reward based off of its action

        :param reward: Reward for its action
        :param is_terminal: If the current simulation is finished (can only be false for tests with continuous simulations)
        """
        raise NotImplemented

    def set_actions(self, actions: iter):
        """
        Available actions for the current scenario

        :param actions:
        """
        self.actions = actions
