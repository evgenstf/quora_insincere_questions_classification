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

def make_prediction(config):
    data_provider = DataProvider(config["data_provider"])
    x_transformer = x_transformer_by_config(config)
    model = model_by_config(config)

    x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
    model.load_train_data(x_transformer.transform(data_provider.x_train), data_provider.y_train)

    prediction = model.predict(x_transformer.transform(data_provider.x_to_predict))
    return prediction

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))

prediction = make_prediction(config)

answer_file = open(config["answer_file"], 'w')
answer_file.write("Id,Probability\n")

for i in range(len(prediction)):
    answer_file.write("%s,%s\n" % (i + 1, prediction[i]))

