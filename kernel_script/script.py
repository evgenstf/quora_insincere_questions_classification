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
from itertools import chain

from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer

from math import sqrt
import spacy
from tqdm import tqdm
import gc


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
    total = 0
    bad_cnt = 0
    for i in range(len(y_predicted)):
        if (y_expected[i] != y_predicted[i]):
            bad_cnt += 1
        total += 1
    return (total - bad_cnt) / total

def false_positive_negative_score(y_expected, y_predicted):
    false_positive = 0
    false_negative = 0
    for i in range(len(y_predicted)):
        if (y_expected[i] != y_predicted[i]):
            if (y_predicted[i] == 1):
                false_negative += 1
            else:
                false_positive += 1
    return false_positive, false_negative


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
        for i in range(len(self.x_known)):
            self.x_known[i] = self.x_known[i].lower()

        self.log.info("loaded {0} x_known lines".format(len(self.x_known)))
        self.log.info("loaded {0} y_known lines".format(len(self.y_known)))

        x_to_predict_file = pd.read_csv(self.x_to_predict_path)
        self.x_to_predict_ids = np.array(x_to_predict_file['qid'].values)
        self.x_to_predict = np.array(x_to_predict_file['question_text'].values)
        for i in range(len(self.x_to_predict)):
            self.x_to_predict[i] = self.x_to_predict[i].lower()
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



#----------EmbeddingProvider----------

from sklearn.model_selection import train_test_split
from random import shuffle
import pandas as pd

class EmbeddingProvider:
    def __init__(self, config):
        self.log = logging.getLogger("EmbeddingProvider")
        self.log.info("embedding provider config: {0}".format(config))
        self.wiki_news_path = config['wiki_news_path']
        self.glove_path = config['glove_path']
        self.paragram_path = config['paragram_path']
        self.porter_stemmer = PorterStemmer()
        self.lancaster_stemmer = LancasterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.log.info("inited")

    def generate_embedding_matrix(self, words, lemmas):
        self.log.info("called")
        def get_coefs(word, *arr):
            print("word:", word, " code: ", words[word])
            return word, np.asarray(arr, dtype='float32')
        
        vector_by_code = dict()
        for string in tqdm(open(self.glove_path)):
            word_and_vector = string.split(" ")
            word = word_and_vector[0]
            vector = word_and_vector[1:]
            if (word in words):
                vector_by_code[words[word]] = vector
            if (word in lemmas):
                word = lemmas[word]
            if (word in words):
                vector_by_code[words[word]] = vector

        self.log.info("vector_by_code inited")
        embed_size = 300
        nb_words = len(words) + 1
        embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
        unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
        self.log.info("start generating embedding_matrix")
        not_vectorized_cnt = 0
        for word in tqdm(words):
            code = words[word]
            if code in vector_by_code:
                embedding_matrix[code] = vector_by_code[code]
            else:
                embedding_matrix[code] = unknown_vector
                not_vectorized_cnt += 1
        self.log.info("not vectorized words count: {}".format(not_vectorized_cnt))
        return embedding_matrix


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
        result = self.vectorizer.transform(x_data).todense()
        self.log.info("transformed")
        result = np.array(result, dtype=np.float16)
        return result

    def features(self):
        return self.vectorizer.get_feature_names()

#----------W2VXTransformer----------

class W2VXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("W2VXTransformer")
        self.log.info("x_transformer config:", config)


        self.log.info("inited")

    def generate_words_and_lemmas(self, textes):
        self.log.info("lemmatize called")
        self.lemmatizer = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
        self.log.info("lemmatizer load")
        self.lemmatizer.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
        self.log.info("lemmatizer flags added")
        self.code_by_word = {}
        word_index = 1
        lemmas = {}
        lemmatized_texts = self.lemmatizer.pipe(textes, n_threads = 2)
        for text in tqdm(lemmatized_texts):
            word_seq = []
            for word in text:
                lemma = word.lemma_
                if (word.text not in self.code_by_word) and (word.pos_ is not "PUNCT"):
                    self.code_by_word[word.text] = word_index
                    word_index += 1
                if (lemma not in self.code_by_word) and (word.pos_ is not "PUNCT"):
                    self.code_by_word[lemma] = word_index
                    word_index += 1
                lemmas[word.text] = word.lemma_
                if word.pos_ is not "PUNCT":
                    word_seq.append(self.code_by_word[lemma])
        del lemmatized_texts
        gc.collect()

        return self.code_by_word, lemmas

    def load_train_data(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = []
        lemmatized_texts = self.lemmatizer.pipe(x_data, n_threads = 2)
        for text in tqdm(lemmatized_texts):
            vector = [0] * len(embedding_matrix[0])
            for word in text:
                if (word.pos_ is not "PUNCT"):
                    embedding_vector = embedding_matrix[self.code_by_word[word.lemma_]]
                    for i in range(len(embedding_vector)):
                        vector[i] += embedding_vector[i]
            result.append(vector)
        return result



#----------x_transformer_by_config----------

def x_transformer_by_config(config):
    x_transormer_config = config["x_transformer"]
    name = x_transormer_config["name"]
    if (name == "dummy"):
        return DummyXTransformer(x_transormer_config)
    if (name == "tfidf"):
        return TfidfXTransformer(x_transormer_config)
    if (name == "w2v"):
        return W2VXTransformer(x_transormer_config)
    logging.fatal("unknown x transformer name: {0}".format(name))
    exit(1)
#----------catboost_model----------

class ClassboostModel:
    def __init__(self, config):
        self.log = logging.getLogger("ClassboostModel")
        self.log.info("model config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostClassifier(
                #logging_level="Silent",
                loss_function=self.config["loss_function"],
                #classes_count=2,
                iterations=self.config["iterations"],
                #l2_leaf_reg=self.config["l2_leaf_reg"],
                learning_rate=self.config["learning_rate"],
                depth=self.config["depth"],
                class_weights=[ 0.1, 0.9 ]
                #thread_count=19
        )
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")

    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(len(x_to_predict)))
        prediction = self.model.predict(x_to_predict)
        self.log.info("predicted")
        return prediction.reshape(-1,)

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
        propabilities = 1 /(1 + np.exp(self.model.decision_function(-x_to_predict)))
        predictions = []
        for propability in propabilities:
            predictions.append(1 if propability > 0.8 else 0)
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
    if (name == "regboost"):
        return RegboostModel(model_config)
    if (name == "classboost"):
        return ClassboostModel(model_config)
    logging.fatal("unknown model name: {0}".format(name))

#----------RegboostSVCModel----------

import catboost as cb
from scipy.stats import spearmanr

class RegboostModel:
    def __init__(self, config):
        self.log = logging.getLogger("RegboostModel")
        self.log.info("model config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostRegressor(
                #logging_level="Silent",
                loss_function=self.config["loss_function"],
                #classes_count=self.config["classes_count"],
                iterations=self.config["iterations"],
                l2_leaf_reg=self.config["l2_leaf_reg"],
                learning_rate=self.config["learning_rate"],
                depth=self.config["depth"],
                #bagging_temperature=self.config["bagging_temperature"],
                metric_period=10,
                thread_count=19,
                random_state=42,
                border_count=100,
                bootstrap_type=config["bootstrap_type"]
                #one_hot_max_size=10
                #one_hot_max_size=self.config["one_hot_max_size"]
        )
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")

    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(len(x_to_predict)))
        prediction = self.round_prediction(self.model.predict(x_to_predict))
        #prediction = self.model.predict(x_to_predict)
        self.log.info("predicted")
        return prediction

    def round_prediction(self, prediction):
        result = [0] * len(prediction)
        for i in range(len(prediction)):
            result[i] = 0 if prediction[i] < -1 else 1
        return result
#----------config----------
config = json.loads("""
{
  "data_provider": {
    "x_known": "../input/train.csv",
    "y_known": "../input/train.csv",
    "x_to_predict": "../input/test.csv",
    "known_using_part" : 1,
    "train_part": 0.5
  },
  "embedding_provider": {
    "glove_path": "../input/embeddings/glove.840B.300d/glove.840B.300d.txt",
    "paragram_path": "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt",
    "wiki_news_path": "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"
  },
  "x_transformer": {
    "name": "w2v"
  },
  "model": {
    "name": "classboost",
    "iterations": 1000,
    "depth": 5,
    "learning_rate": 0.3,
    "l2_leaf_reg":0.07,
    "loss_function": "MultiClassOneVsAll",
    "classes_count": 2
  },
  "model2": {
    "name": "regboost",
    "iterations": 100,
    "depth": 8,
    "learning_rate": 0.2,
    "l2_leaf_reg":0.07,
    "loss_function": "RMSE",
    "classes_count": 5,
    "bootstrap_type" : "No"
  },
  "answer_file": "submission.csv"
}
""")
#----------Launcher----------

def calculate_and_print_test_score(data_provider, x_transformer, model, config):
    test_prediction = model.predict(x_transformer.transform(data_provider.x_test))
    false_positive, false_negative = false_positive_negative_score(data_provider.y_test, test_prediction)
    total_positive = 0
    total_negative = 0
    predicted_positive = 0
    predicted_negative = 0
    for i in range(len(data_provider.y_test)):
        x = data_provider.y_test[i]
        if x == 1:
            total_negative += 1
            if (x == test_prediction[i]):
                predicted_negative += 1
        else:
            total_positive += 1
            if (x == test_prediction[i]):
                predicted_positive += 1
    acc_positive = predicted_positive / (predicted_positive + false_positive)
    acc_negative = predicted_negative / (predicted_negative + false_negative)
    acc = (acc_negative + acc_positive) / 2

    rec_positive = predicted_positive / (predicted_positive + false_negative)
    rec_negative = predicted_negative / (predicted_negative + false_positive)
    rec = (rec_negative + rec_positive) / 2

    f_score = (rec * acc) / (rec + acc) * 2
    print("************************")
    print("predicted_positive:", predicted_positive, "/", total_positive)
    print("predicted_negative:", predicted_negative, "/", total_negative)
    print("F score: ", f_score)
    print("************************")
    score_file = open(str(f_score), 'w')
    score_file.write(str(json.dumps(config)))

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))


data_provider = DataProvider(config["data_provider"])

x_transformer = x_transformer_by_config(config)
code_by_word, lemma_by_word = x_transformer.generate_words_and_lemmas(np.concatenate((data_provider.x_known, data_provider.x_to_predict)))

embedding_provider = EmbeddingProvider(config["embedding_provider"])
embedding_matrix = embedding_provider.generate_embedding_matrix(code_by_word, lemma_by_word)

log.info("embedding_matrix generated with shape: {}".format(embedding_matrix.shape))


x_transformer.load_train_data(embedding_matrix)

model = model_by_config(config)
model.load_train_data(x_transformer.transform(data_provider.x_train), data_provider.y_train)

calculate_and_print_test_score(data_provider, x_transformer, model, config)

prediction = model.predict(x_transformer.transform(data_provider.x_to_predict))

answer_file = open(config["answer_file"], 'w')
answer_file.write("qid,prediction\n")

for i in range(len(prediction)):
    answer_file.write("%s,%s\n" % (data_provider.x_to_predict_ids[i], int(prediction[i])))

