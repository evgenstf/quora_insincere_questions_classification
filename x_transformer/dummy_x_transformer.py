import sys
sys.path.append("../../base")
from common import *

#----------DummyXTransformer----------

class DummyXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("DummyXTransformer")
        self.log.info("x_transformer config:", config)
        self.config = config
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.x_train = x_train
        self.y_train = y_train
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = x_data
        self.log.info("transformed")
        return result
