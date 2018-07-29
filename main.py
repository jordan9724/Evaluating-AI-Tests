# from models.mnist import CNN
# from models.replay import CNNReplay
# from runner.runner import TestRunner
# from tools.setup import setup_tensorflow

# setup_tensorflow()

# TestRunner(CNNReplay).run_tests()
import ipdb
import numpy as np

from tools.analysis import DataAnalyzer
from math import e


def get_intelligence(epochs: int, scores: iter):
    def exp_decay(x: int):
        return e ** (-x / epochs)

    total_decay = sum([exp_decay(x) for x in range(epochs)])
    score = sum([float(scores[s] - scores[s - 1]) * float(exp_decay(s)) / float(total_decay) for s in range(1, epochs)])
    return score


handicaps = [1., 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.]
tests = ['MNIST', 'Doom Basic']
models = ['CNN', 'Replay CNN']
intelligence_scores = {
    model: {
        test: {
            handicap: []
            for handicap in handicaps
        }
        for test in tests
    }
    for model in models
}

epochs = []
for d_id, model in zip([26, 27], models):
    self = DataAnalyzer(data_id=d_id)
    for test in tests:
        test_df = self._df.loc[self._df['Test'] == test]
        epochs = test_df.loc[:, 'Epoch'].unique()
        min_val = test_df.loc[:, 'Min'].mean()
        max_val = test_df.loc[:, 'Max'].mean()
        y_vals_score = {handicap: [] for handicap in handicaps}
        for epoch in epochs:
            for handicap in handicaps:
                nums = test_df.loc[test_df['Epoch'] == epoch].loc[test_df['Handicap'] == handicap]
                y_vals_score[handicap].append((nums['Score'].sum() - min_val) / (max_val - min_val))
        for handicap in handicaps:
            intelligence_scores[model][test][handicap] = get_intelligence(len(epochs), y_vals_score[handicap])

# da = DataAnalyzer(len(models) * len(tests) * len(handicaps), ['Model', 'Test', 'Handicap', 'Score'])
# for model in models:
#     for test in tests:
#         for handicap in handicaps:
#             da.set_next_values([model, test, handicap, intelligence_scores[model][test][handicap]])
#
# da.display_intelligence_score()


def get_overall_intelligence(_handicaps, int_scores):
    assert len(_handicaps) == len(int_scores)

    def lin_decay(x):
        return (-x / 2) + 1

    total_decay = sum([lin_decay(x / len(_handicaps)) for x in range(len(_handicaps))])
    return sum([int_scores[x] * lin_decay(x / len(int_scores)) / total_decay for x in range(len(int_scores))])


for model in models:
    for test in tests:
        scores = []
        for handicap in handicaps[::-1]:
            scores.append(np.mean(intelligence_scores[model][test][handicap]))
        print('Model:', model, 'Test:', test, 'Score:', get_overall_intelligence(handicaps, scores))
