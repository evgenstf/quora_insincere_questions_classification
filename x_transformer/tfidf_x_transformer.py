import sys
sys.path.append("../../base")
from common import *
import string

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
        result = self.vectorizer.transform(x_data)
        self.log.info("transformed")
        return result

    def features(self):
        return self.vectorizer.get_feature_names()
