from random import randint

from models.base import ModelBase


class FirstAction(ModelBase):

    def get_action(self, **kwargs):
        return 0


class RandAction(ModelBase):

    def get_action(self, **kwargs):
        return randint(0, self.test_info.num_actions - 1)
