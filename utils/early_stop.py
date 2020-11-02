import numpy as np
from utils import global_var as glb


class EarlyStopping:
    def __init__(self, min_delta, improve_range, score_type):
        """
        :param target: glb.cv_train or glb.cv_valid
            Choose which score is used in early stopping, either train or valid
        :param min_delta: float
            Minimum required improvement during improve_range
        :param improve_range: int
            Define the length of epochs that score_type need to be improved more than min_delta
        :param score_type: str
            Either "loss" or "acc"
        """
        # initial variables
        self.min_delta = min_delta
        self.improve_range = improve_range
        assert score_type in ["loss", "acc"]
        self.score_type = score_type
        # set later
        self.hist_scores = list()  # history of scores per epoch
        self.is_stop = False

    def __str__(self):
        return f"min_delta: {self.min_delta}, improve_range: {self.improve_range}, " \
               f"score_type: {self.score_type}, hist: {self.hist_scores}"

    def __get_score(self, loss, acc):
        return loss if self.score_type == "loss" else acc

    def __is_improved(self, base, comp_values):
        if self.score_type == "loss":
            return base > np.min(comp_values)
        else:
            return base < np.max(comp_values)

    def set_stop_flg(self, loss=0, acc=0):
        score = self.__get_score(loss, acc)
        self.hist_scores.append(score)

        if len(self.hist_scores) > self.improve_range:
            cur_value = self.hist_scores[-self.improve_range - 1]
            comp_values = self.hist_scores[-self.improve_range:]
            is_improved = self.__is_improved(cur_value, comp_values)
            self.is_stop = not is_improved
