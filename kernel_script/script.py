import json
import warnings
warnings.filterwarnings("ignore")

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO)

#ch = logging.StreamHandler(sys.stdout)
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s %(levelname)s[%(name)s] - %(message)s')
#ch.setFormatter(formatter)
#root.addHandler(ch)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from math import sqrt



def draw_pair_plot(x_data, y_data):
    ncol, nrow = 7, x_data.shape[1] // 7 + (x_data.shape[1] % 7 > 0)
    plt.figure(figsize=(ncol * 4, nrow * 4))

    for i, feature in enumerate(x_data.columns):
        plt.subplot(nrow, ncol, i + 1)
        plt.scatter(x_data[feature], y_data, s=10, marker='o', alpha=.6)
        plt.xlabel(feature)
        if i % ncol == 0:
            plt.ylabel('target')

def mape_score(y_data, prediction): 
    total = 0
    bad_cnt = 0
    for i in range(len(y_data.as_matrix())):
        loss_value = np.abs((y_data.as_matrix()[i][0] - prediction[i]) / (y_data.as_matrix()[i][0]))
        if (loss_value > 0.08):
            print("loss_value:", loss_value, "index:", i)
            bad_cnt += 1
        total += loss_value
    print("bad_cnt:", bad_cnt)
    total /= len(y_data)
    total = total * 100
    return total

def ratio_score(y_expected, y_predicted):
    return roc_auc_score(y_expected[:len(y_predicted)], y_predicted)

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

#----------TfidfXTransformer----------

from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("TfidfXTransformer")
        self.log.info("x_transformer config:", config)

        self.ngram_range = config["ngram_range"]
        self.min_df = config["min_df"]
        self.max_df = config["max_df"]
        self.max_features = config["max_features"]

        self.vectorizer = TfidfVectorizer(
                min_df = self.min_df,
                max_df = self.max_df,
                ngram_range = self.ngram_range,
                lowercase = True,
                sublinear_tf = True,
                # tokenizer=tokenize,
                # stop_words = ["must", "and"]
                # stop_words = stopwords.words('english')
                norm='l2',
                max_features = self.max_features
        )

        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.vectorizer.fit(x_train)
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = self.vectorizer.transform(x_data)
        self.log.info("transformed")
        return result

    def features(self):
        return self.vectorizer.get_feature_names()

#----------x_transformer_by_config----------

def x_transformer_by_config(config):
    x_transormer_config = config["x_transformer"]
    name = x_transormer_config["name"]
    if (name == "dummy"):
        return DummyXTransformer(x_transormer_config)
    if (name == "tfidf"):
        return TfidfXTransformer(x_transormer_config)
    logging.fatal("unknown x transformer name: {0}".format(name))
    exit(1)

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
        predictions = 1 /(1 + np.exp(self.model.decision_function(-x_to_predict)))
        return predictions

    def weights(self):
        return [0]

#----------model_by_config----------

def model_by_config(config):
    model_config = config["model"]
    name = model_config["name"]
    if (name == "dummy"):
        return DummyModel(model_config)
    if (name == "linear_svc"):
        return LinearSVCModel(model_config)
    logging.fatal("unknown model name: {0}".format(name))
#----------config----------
config = json.loads("""
{
  "data_provider": {
    "x_known": "../input/head_train.csv",
    "y_known": "../input/head_train.csv",
    "x_to_predict": "../input/head_test.csv",
    "known_using_part" : 0.01,
    "train_part": 0.99999999999
  },
  "x_transformer": {
    "name": "tfidf",
    "min_df": 0,
    "max_df": 0.9,
    "ngram_range" : [1, 2],
    "max_features": 100000000
  },
  "model": {
    "name": "linear_svc"
  },
  "answer_file": "submission.csv"
}
""")

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))

data_provider = DataProvider(config["data_provider"])

x_transformer = x_transformer_by_config(config)
model = model_by_config(config)

x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
model.load_train_data(x_transformer.transform(data_provider.x_train), data_provider.y_train)

prediction = model.predict(x_transformer.transform(data_provider.x_to_predict))


answer_file = open(config["answer_file"], 'w')
answer_file.write("qid,prediction\n")

for i in range(len(prediction)):
    answer_file.write("%s,%s\n" % (data_provider.x_to_predict_ids[i], prediction[i]))

