import sys
sys.path.append("../base")
sys.path.append("../data_provider")
sys.path.append("../x_transformer")
sys.path.append("../model")

from common import *
from data_provider import *
from x_transformer_by_config import *
from model_by_config import *

if (len(sys.argv) < 2):
    print("Usage: ./example.py <config>")
    exit()

config_file = open(sys.argv[1], 'r')
config = json.load(config_file)

#----------Launcher----------

def calculate_and_print_test_score(data_provider, x_transformer, model, config):
    test_prediction = model.predict(x_transformer.transform(data_provider.x_test))
    test_score = ratio_score(test_prediction, data_provider.y_test)
    print("************************")
    print("test_score:", test_score)
    print("************************")
    score_file = open('scores/' + str(test_score), 'w')
    score_file.write(str(json.dumps(config)))

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))

data_provider = DataProvider(config["data_provider"])

x_transformer = x_transformer_by_config(config)
x_transformer.load_train_data(data_provider.x_known, data_provider.y_known)

exit()

model = model_by_config(config)

x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
model.load_train_data(x_transformer.transform(data_provider.x_train), data_provider.y_train)

calculate_and_print_test_score(data_provider, x_transformer, model, config)

prediction = model.predict(x_transformer.transform(data_provider.x_to_predict))

answer_file = open(config["answer_file"], 'w')
answer_file.write("qid,prediction\n")

for i in range(len(prediction)):
    answer_file.write("%s,%s\n" % (data_provider.x_to_predict_ids[i], prediction[i]))

