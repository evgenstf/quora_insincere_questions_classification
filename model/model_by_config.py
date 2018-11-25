import sys
sys.path.append("../../base")
from common import *

from dummy_model import *

def model_by_config(config):
    model_config = config["model"]
    name = model_config["name"]
    if (name == "dummy"):
        return DummyModel(model_config)
    logging.fatal("unknown model name: {0}".format(name))
