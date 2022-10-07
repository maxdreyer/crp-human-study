import sys

import torch
import torch.hub
from torchvision.models import vgg16, vgg13_bn, alexnet


def get_vgg16(ckpt_path=None, **kwargs) -> torch.nn.Module:
    model = vgg16(pretrained=True)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    return model


def get_vgg16_corrupted(
        ckpt_path="models/ckpts/vgg16p_classification_imagenet_corrupted.pth",
        **kwargs) -> torch.nn.Module:
    model = vgg16(pretrained=True)
    if ckpt_path:
        print("loading checkpoint", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
    return model


def get_vgg16_uncorrupted(
        ckpt_path="models/ckpts/vgg16p_classification_imagenet_padded.pth",
        **kwargs) -> torch.nn.Module:
    model = vgg16(pretrained=True)
    if ckpt_path:
        print("loading checkpoint", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
    return model


def get_alexnet(ckpt_path=None, **kwargs) -> torch.nn.Module:
    model = alexnet(pretrained=True)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    return model
