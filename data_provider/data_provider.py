import sys
sys.path.append("../base")
from common import *

#----------DataProvider----------

from sklearn.model_selection import train_test_split
from random import shuffle
import pandas as pd

class DataProvider:
    def __init__(self, config):
        self.log = logging.getLogger("DataProvider")
        self.log.info("data provider config: {0}".format(config))
        self.x_known_path = config["x_known"]
        self.y_known_path = config["y_known"]
        self.x_to_predict_path = config["x_to_predict"]
        self.known_using_part = config["known_using_part"]

        x_known_file = open(self.x_known_path, 'r')
        self.x_known = np.array(pd.read_csv(self.x_known_path)['question_text'].values)
        self.y_known = np.array(pd.read_csv(self.y_known_path)['target'].values)

        known_using_count = int(len(self.x_known) * self.known_using_part)
        self.x_known = self.x_known[:known_using_count]
        self.y_known = self.y_known[:known_using_count]

        self.log.info("loaded {0} x_known lines".format(len(self.x_known)))
        self.log.info("loaded {0} y_known lines".format(len(self.y_known)))

        x_to_predict_file = pd.read_csv(self.x_to_predict_path)
        self.x_to_predict_ids = np.array(x_to_predict_file['qid'].values)
        self.x_to_predict = np.array(x_to_predict_file['question_text'].values)
        self.log.info("loaded {0} x_to_predict lines".format(len(self.x_to_predict)))


        self.split_known_data_to_train_and_test(config["train_part"])

        self.log.info("inited")

    def split_known_data_to_train_and_test(self, train_part):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_known, self.y_known, train_size = train_part, random_state = 42)
        self.log.info("splitted known data: "
                "x_train size: {0} y_train size: {1} x_test size: {2} y_test size: {3}".format(
                        len(self.x_train), len(self.y_train), len(self.x_test), len(self.y_test)
                )
        )


