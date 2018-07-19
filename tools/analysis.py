import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from slugify import slugify
from tools.save_file import SaveInfo


class DataAnalyzer:

    def __init__(self, num_trials: int, columns: iter, data_id=None):
        self._saver_info = SaveInfo('data', data_id, 'data.json')
        if data_id is not None:
            self._curr_index = num_trials
            self._df = self.load()
        else:
            self._curr_index = 0
            self._df = pd.DataFrame(np.empty([num_trials, len(columns)]), index=np.arange(num_trials), columns=columns)

    def set_next_values(self, vals: iter):
        assert len(vals) == len(self._df.columns)
        assert self._curr_index < len(self._df), "Did you load this data? It's not recommended you change data from a previous trial"
        self._df.loc[self._curr_index] = vals
        self._curr_index += 1

    @staticmethod
    def get_ticks(skip, vals):
        """
        Returns a range of ticks (for x and y axis) that encapsulate values in `vals`

        min_tick is a skip multiple below the min vals value
        max_tick is a skip multiple above the max vals value

        EXAMPLE::
            skip = 50
            vals = [-159.5, -30.7, 12.2, 51.8]
                => min_tick = -200
                => max_tick = 150
                => np.arange will make a list equivalent to [-200, -150, -100, -50, 0, 50, 100]
        """
        min_val = int(min(vals))
        max_val = int(max(vals) + skip * 0.999)

        min_tick = min_val - min_val % skip
        max_tick = max_val + skip - max_val % skip
        return np.arange(min_tick, max_tick, skip)

    def display_error_bar(self, x_axis: str, y_axis: str, title, x_label=None, y_label=None, x_skip=None, y_skip=None):
        x_vals = self._df.loc[:, 'Epoch'].unique()
        y_vals = [self._df.loc[self._df['Epoch'] == i].loc[:, 'Score'].mean() for i in x_vals]
        error = [self._df.loc[self._df['Epoch'] == i].loc[:, 'Score'].std() for i in x_vals]

        plt.errorbar(x_vals, y_vals, yerr=error, fmt='.-')

        # Applies optional parameters
        if x_skip:
            plt.xticks(self.get_ticks(x_skip, x_vals))
        if y_skip:
            # Puts the minimum and maximum values possible created from the error bars into `y_vals_range`
            lower_y = np.array(y_vals) - np.array(error)
            upper_y = np.array(y_vals) + np.array(error)
            y_vals_range = np.concatenate((lower_y, upper_y))
            plt.yticks(self.get_ticks(y_skip, y_vals_range))

        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

        plt.grid(True)
        plt.savefig(SaveInfo('graphs', self._saver_info.save_num, '{}.png'.format(slugify(title).replace('-', '_'))).get_file_name())
        plt.show()

    def save(self):
        with open(self._saver_info.get_file_name(), 'w') as outfile:
            json.dump(self._df.to_json(orient='split'), outfile)

    def load(self):
        with open(self._saver_info.get_file_name(), 'r') as infile:
            return pd.read_json(json.load(infile), orient='split')
