import sys
sys.path.append("../base")
sys.path.append("../data_provider")
sys.path.append("../x_transformer")
sys.path.append("../model")

from common import *
from data_provider import *
from embedding_provider import *
from x_transformer_by_config import *
from model_by_config import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

config_file = open(sys.argv[1], 'r')
config = json.load(config_file)

#----------Launcher----------

def calculate_and_print_test_score(data_provider, x_transformer, model, config):
    test_prediction = model.predict(x_transformer.transform(data_provider.x_known))
    false_positive, false_negative = false_positive_negative_score(data_provider.y_known, test_prediction)
    total_positive = 0
    total_negative = 0
    predicted_positive = 0
    predicted_negative = 0
    for i in range(len(data_provider.y_known)):
        x = data_provider.y_known[i]
        if x == 1:
            total_negative += 1
            if (x == test_prediction[i]):
                predicted_negative += 1
        else:
            total_positive += 1
            if (x == test_prediction[i]):
                predicted_positive += 1
    print("************************")
    print("predicted_positive:", predicted_positive, "/", total_positive)
    print("predicted_negative:", predicted_negative, "/", total_negative)
    positive_acc = predicted_positive / total_positive
    negative_acc = predicted_negative / total_negative
    print("F score: ", (positive_acc * negative_acc) / (positive_acc + negative_acc) * 2)
    print("************************")
    #score_file = open('scores/' + str(false_positive) + '_' + str(false_negative), 'w')
    #score_file.write(str(json.dumps(config)))

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

