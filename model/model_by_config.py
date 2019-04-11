import sys
sys.path.append("../../base")
from common import *

from dummy_model import *
from linear_svc_model import *
from regboost_model import *



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
    logging.fatal("unknown model name: {0}".format(name))
