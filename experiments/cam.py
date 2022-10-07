import os

import click
import torch.cuda
import zennit
from PIL import Image
from captum.attr import ShapleyValueSampling, GuidedGradCam, LayerGradCam, LayerAttribution
from fast_slic import Slic
from torchvision.models import vgg16

from datasets_ import get_dataset, get_sample
import matplotlib.pyplot as plt
import zennit.image as zimage

from models import get_model


@click.command()
@click.option("--model_name", default="vgg16")
@click.option("--dataset_name", default="imagenet")
@click.option("--sample", default=1111)
@click.option("--layer_name", default="features.26")
def cam(model_name, dataset_name, sample, layer_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device)
    model.eval()
    replace_relus(model)
    dataset = get_dataset(dataset_name)["test"]()

    data, target = get_sample(dataset, sample_id=sample, device=device)

    layer = [m for n, m in model.named_modules() if layer_name in n]

    if len(layer) == 0:
        print("layer not found")
    elif len(layer) > 1:
        print("more than one layer found")

    ggc = LayerGradCam(model, layer[0])

    attr = ggc.attribute(data.requires_grad_(), target=target.long().item())
    upsampled_attr = LayerAttribution.interpolate(attr, (224, 224))
    heatmap = zimage.imgify(upsampled_attr.detach().cpu().sum(1), symmetric=True, cmap="bwr")

    os.makedirs(f"results/cam/{dataset_name}", exist_ok=True)
    heatmap.save(f"results/cam/{dataset_name}/sample_{sample}_{model_name}_cam.png")


def replace_relus(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_relus(module)
        if isinstance(module, torch.nn.ReLU):
            new = torch.nn.ReLU()
            setattr(model, n, new)
            print(f"Replaced Relu with Relu inplace false.")

if __name__ == "__main__":
    cam()
