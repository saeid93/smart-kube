# TODO change it to object foctory
#      e.g.
#      https://realpython.com/factory-method-python/#a-generic-interface-to-object-factory
from .constants import ENVS

def make_env(type_env, env_config):
    """
    make the environment object
    """
    return make_env_class(type_env)(env_config)


def make_env_class(type_env):
    """
    generate the class object
    """
    return ENVS[type_env]
