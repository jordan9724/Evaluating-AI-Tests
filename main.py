from models.cnn import CNN
from runner.runner import TestRunner
from tools.analysis import DataAnalyzer
from tools.setup import setup_tensorflow

setup_tensorflow()

TestRunner(CNN).start()
