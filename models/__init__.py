import torch

from models.adience_model import get_vgg16_bn_adience, get_vgg16_adience, get_vgg16_adience_lfp
from models.resnet import get_resnet34
from models.vgg16 import get_vgg16, get_vgg16_corrupted, get_vgg16_uncorrupted, \
    get_alexnet

MODELS = {
    "vgg16": get_vgg16,
    "vgg16_corrupted": get_vgg16_corrupted,
    "vgg16_uncorrupted": get_vgg16_uncorrupted,
    "alexnet": get_alexnet,
}


def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    try:
        model = MODELS[model_name](**kwargs)
        return model
    except KeyError:
        print(f"Model {model_name} not available")
        exit()
