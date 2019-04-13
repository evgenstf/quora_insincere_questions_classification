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
        lemmatizer = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
        self.log.info("lemmatizer load")
        lemmatizer.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
        self.log.info("lemmatizer flags added")
        words = {}
        word_index = 1
        lemmas = {}
        lemmatized_texts = lemmatizer.pipe(textes, n_threads = 2)
        word_sequences = []
        for text in tqdm(lemmatized_texts):
            word_seq = []
            for word in text:
                if (word.text not in words) and (word.pos_ is not "PUNCT"):
                    words[word.text] = word_index
                    word_index += 1
                    lemmas[word.text] = word.lemma_
                if word.pos_ is not "PUNCT":
                    word_seq.append(words[word.text])
            word_sequences.append(word_seq)
        del lemmatized_texts
        gc.collect()
        return word_sequences, words, lemmas

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        lemmatized_sequences = self.lemmatize(x_data)
        result, words, lemmas = generate_words_and_lemmas(x_data)
        return result

