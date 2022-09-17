# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


_model_entrypoints = {}


def register_model(fn):
    model_name = fn.__name__
    _model_entrypoints[model_name] = fn
    return fn


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def create_model(model_name, pretrained, **kwargs):
    create_fn = _model_entrypoints[model_name]
    model = create_fn(pretrained, **kwargs)
    return model

