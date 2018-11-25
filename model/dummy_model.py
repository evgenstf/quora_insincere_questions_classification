import sys
sys.path.append("../../base")
from common import *

#----------DummyModel----------

class DummyModel:
    def __init__(self, config):
        self.log = logging.getLogger("SkLearnCountVectorizerModel")
        self.log.info("model config:", config)
        self.config = config
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(x_train.shape[0], len(y_train)))
        self.x_train = x_train
        self.y_train = y_train
        self.log.info("loaded")

    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(x_to_predict.shape[0]))
        result = [1]
        self.log.info("predicted")
        return result
