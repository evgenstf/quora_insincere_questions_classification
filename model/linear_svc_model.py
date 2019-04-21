import sys
sys.path.append("../../base")
from common import *







#----------LinearSVCModel----------

from sklearn.feature_extraction import text
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

class LinearSVCModel:
    def __init__(self, config):
        self.log = logging.getLogger("LinearSVCModel")
        self.name = "linear_svc"
        self.model = LinearSVC()
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(x_train.shape[0], len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")


    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(x_to_predict.shape[0]))
        propabilities = 1 /(1 + np.exp(self.model.decision_function(-x_to_predict)))
        predictions = []
        for propability in propabilities:
            predictions.append(1 if propability > 0.8 else 0)
        return predictions

    def weights(self):
        return [0]
