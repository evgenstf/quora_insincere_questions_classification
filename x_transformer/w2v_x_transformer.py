import sys
sys.path.append("../../base")
from common import *
import string






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

