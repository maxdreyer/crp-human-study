import os

import click
import torch.cuda
import zennit
from PIL import Image
from captum.attr import ShapleyValueSampling
from fast_slic import Slic
from datasets_ import get_dataset, get_sample
import zennit.image as zimage

from models import get_model


@click.command()
@click.option("--model_name", default="vgg16")
@click.option("--dataset_name", default="imagenet")
@click.option("--sample", default=1111)
def shap(model_name, dataset_name, sample):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device)
    model.eval()
    dataset = get_dataset(dataset_name)["test"]()

    data, target = get_sample(dataset, sample_id=sample, device=device)

    slic = Slic(num_components=100, compactness=10)
    img = dataset.reverse_augmentation(data[0]).permute((1, 2, 0)).detach().cpu().numpy().astype("uint8")
    assignment = torch.tensor(slic.iterate(img.copy(order='C'))).to(device)

    svs = ShapleyValueSampling(model)
    attr = svs.attribute(data, target=target.long().item(), n_samples=50, feature_mask=assignment, show_progress=True,
                         baselines=0.5)
    heatmap = zimage.imgify(attr.detach().cpu().sum(1), symmetric=True, cmap="bwr")

    os.makedirs(f"results/shap/{dataset_name}", exist_ok=True)
    heatmap.save(f"results/shap/{dataset_name}/sample_{sample}_{model_name}_shap.png")

if __name__ == "__main__":
    shap()
