from runner.runner import TestInfo


class ModelBase:
    test_info = None

    def set_test_info(self, test_info: TestInfo):
        """
        Available actions for the current scenario

        :param test_info: Contains information about the current state of the test
        """
        self.test_info = test_info

    def get_action(self):
        """
        Gives back an action with assistance of the given data
        :return: The chosen action index
        """
        raise NotImplemented

    def receive_reward(self):
        """
        After the action is received, the model will be given a reward based off of its action
        """
        pass

    def save_model(self):
        """
        Called to suggest the model to save
        """
        pass

    def load_model(self):
        """
        Restores a model
        """
        pass
