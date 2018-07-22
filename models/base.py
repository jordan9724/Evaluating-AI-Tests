class ModelBase:
    num_actions = 0

    def set_num_actions(self, actions: iter):
        """
        Available actions for the current scenario

        :param actions:
        """
        self.num_actions = actions

    def get_action(self, is_training: bool, data, epoch=None, max_epoch=None):
        """
        Gives back an action with assistance of the given data

        :param is_training: True when the system should be learning
        :param data: Used to determine an action
        :param epoch: Current epoch
        :param max_epoch: Max number of epochs
        :return: The chosen action
        """
        raise NotImplemented

    def receive_reward(self, is_training: bool, data, reward: float, is_terminal: bool):
        """
        After the action is received, the model will be given a reward based off of its action

        :param is_training: True when the system should be learning
        :param data: Data of the state after learning (None if test is not sequential)
        :param reward: Reward for its action
        :param is_terminal: If the current simulation is finished (can only be false for tests with continuous simulations)
        """
        pass
