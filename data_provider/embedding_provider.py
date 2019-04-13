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
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(self.glove_path) if len(o) > 100)
        self.log.info("indexes loaded")
        embed_size = 300
        nb_words = len(words) + 1
        embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
        unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
        for key in tqdm(words):
            self.log.info("process word: {}".format(key))
            word = key
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            word = key.lower()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            word = key.upper()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            word = key.capitalize()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            word = self.porter_stemmer.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            word = self.lancaster_stemmer.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            word = self.snowball_stemmer.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            word = lemmas[key]
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[words[key]] = embedding_vector
                continue
            embedding_matrix[words[key]] = unknown_vector                    
        return 0

