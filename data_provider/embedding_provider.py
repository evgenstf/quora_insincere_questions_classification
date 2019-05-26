import sys
sys.path.append("../base")
from common import *







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

