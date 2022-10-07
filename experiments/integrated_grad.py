import os

import click
import torch.cuda
import zennit
from PIL import Image
from captum.attr import ShapleyValueSampling, IntegratedGradients
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
def integrated_gradient(model_name, dataset_name, sample):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device)
    model.eval()
    dataset = get_dataset(dataset_name)["test"]()

    data, target = get_sample(dataset, sample_id=sample, device=device)

    ig = IntegratedGradients(model)
    attr = ig.attribute(data, target=target.long().item(), n_steps=100)
    heatmap = zimage.imgify(attr.detach().cpu().sum(1), symmetric=True, cmap="bwr")

    os.makedirs(f"results/integrated_grad/{dataset_name}", exist_ok=True)
    heatmap.save(f"results/integrated_grad/{dataset_name}/sample_{sample}_{model_name}_integrated_grad.png")

if __name__ == "__main__":
    integrated_gradient()
