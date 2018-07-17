import pandas as pd


class DataAnalyzer:

    def __init__(self, num_trials: int, columns: iter):
        self._df = pd.DataFrame(index=range(num_trials), columns=columns)
        self._curr_index = 0

    def set_next_values(self, vals: iter):
        assert len(vals) == len(self._df)
        assert self._curr_index < len(self._df.columns)

        self._df.loc[:, self._curr_index] = vals
        self._curr_index += 1
