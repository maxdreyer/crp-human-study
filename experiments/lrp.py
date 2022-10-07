import os

import click
import numpy as np
import torch.cuda
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from torchvision.models import vgg16

from datasets_ import get_dataset, get_sample
import torchvision.transforms as T
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
import zennit.image as zimage

from models import get_model


@click.command()
@click.option("--model_name", default="vgg16")
@click.option("--dataset_name", default="imagenet")
@click.option("--sample", default=64)
def crp(model_name, dataset_name, sample):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device)
    model.eval()

    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
    dataset = get_dataset(dataset_name)["test"]()

    attribution = CondAttribution(model)
    data, target = get_sample(dataset, sample_id=sample, device=device)
    attr = attribution(data.requires_grad_(), [{"y": target.long().item()}], composite)
    attr = attr.heatmap
    heatmap = zimage.imgify(attr.detach().cpu(), symmetric=True, cmap="bwr")
    os.makedirs(f"results/lrp/{dataset_name}", exist_ok=True)
    heatmap.save(f"results/lrp/{dataset_name}/sample_{sample}_{model_name}_lrp.png")





if __name__ == "__main__":
    crp()
