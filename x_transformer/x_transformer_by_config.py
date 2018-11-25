import sys
sys.path.append("../../base")
from common import *

from dummy_x_transformer import *

def x_transformer_by_config(config):
    x_transormer_config = config["x_transformer"]
    name = x_transormer_config["name"]
    if (name == "dummy"):
        return DummyXTransformer(x_transormer_config)
    logging.fatal("unknown x transformer name: {0}".format(name))
