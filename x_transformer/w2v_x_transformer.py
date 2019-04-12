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

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.x_train = x_train
        self.y_train = y_train
        self.lemmatize()
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = self.vectorizer.transform(x_data).todense()
        self.log.info("transformed")
        result = np.array(result, dtype=np.float16)
        return result

    def features(self):
        return self.vectorizer.get_feature_names()

    def lemmatize(self):
        self.log.info("lemmatize called")
        lemmatizer = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
        self.log.info("lemmatizer load")
        lemmatizer.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
        self.log.info("lemmatizer flags added")
        word_dict = {}
        word_index = 1
        lemma_dict = {}
        docs = lemmatizer.pipe(self.x_train, n_threads = 2)
        word_sequences = []
        for doc in tqdm(docs):
            word_seq = []
            for token in doc:
                if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
                    word_dict[token.text] = word_index
                    word_index += 1
                    lemma_dict[token.text] = token.lemma_
                if token.pos_ is not "PUNCT":
                    word_seq.append(word_dict[token.text])
            word_sequences.append(word_seq)
        del docs
        gc.collect()
        #train_word_sequences = word_sequences[:num_train_data]
        #test_word_sequences = word_sequences[num_train_data:]

