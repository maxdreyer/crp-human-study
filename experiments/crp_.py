import os
from pathlib import Path
from typing import Any, Dict

import click
import numpy as np
import pandas as pd
import torch.cuda
import torchvision
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid
from zennit.torchvision import ResNetCanonizer

from datasets_ import get_dataset, get_sample
import torchvision.transforms as T
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
import zennit.image as zimage
from scipy.ndimage import gaussian_filter

from models import get_model
from utils.crp_utils import ReceptiveFieldCRP, CondAttributionDiff
from utils.milan import milan

@click.command()
@click.option("--model_name", default="vgg_adience_lfp")
@click.option("--dataset_name", default="adience")
@click.option("--sample", default=0)
@click.option("--layer_name", default="features.28")
@click.option("--mode", default="relevance")
@click.option("--neuron_indices", default=None)
@click.option("--class_specific_samples", default=False, type=bool)
def crp_plot(model_name, dataset_name, sample, layer_name, mode, neuron_indices, class_specific_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device)
    model.eval()

    decoder = milan.pretrained('base+clip').to(device)

    canonizers = [ResNetCanonizer()]
    composite = EpsilonPlusFlat(canonizers)
    dataset = get_dataset(dataset_name)["test"](preprocessing=False)

    cc = ChannelConcept()

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer: cc for layer in layer_names}
    attribution = CondAttributionDiff(model)

    data, target = get_sample(dataset, sample_id=sample, device=device)
    target = target.long().item()
    # separate normalization from resizing for plotting purposes later
    preprocessing = dataset.preprocessing

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing,
                              path=f"{model_name}_{dataset_name}")
    rf = ReceptiveFieldCRP(attribution, data.requires_grad_(), path=f"{model_name}_{dataset_name}")
    if not (rf.PATH / Path(f"{layer_name}.npy")).is_file():
        rf.run({layer_name: cc}, canonizers=canonizers, batch_size=32)
    fv.add_receptive_field(rf)
    attr = attribution(preprocessing(data.requires_grad_()),
                       [{"y": target}],
                       composite,
                       record_layer=[layer_name], init_rel=1)
    # print(torch.topk(attr.prediction[0], min(5, len(attr.prediction[0]))), target)
    channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)
    if neuron_indices is None:
        topk_c = 5
        topk = torch.topk(channel_rels[0].abs(), topk_c).indices.detach().cpu().numpy()
        topk_rel = channel_rels[0][topk]
    else:
        topk = [int(x) for x in neuron_indices.split(",")]
        topk_rel = channel_rels[0][topk].detach().cpu().numpy()
        topk_c = len(topk)

    print("most relevant neurons:")
    print(topk, topk_rel)
    conditions = [{"y": target, layer_name: c} for c in topk]
    heatmaps, _, _, _ = attribution(preprocessing(data.requires_grad_()), conditions, composite)

    n_refimgs = 14
    RF = True
    if class_specific_samples:
        ref_imgs = {}
        for c in topk:
            ref_imgs[c] = fv.get_stats_reference(c, layer_name, [target], mode, (0, n_refimgs), rf=RF)[target]
    else:
        ref_imgs = fv.get_max_reference(topk, layer_name, mode, (0, n_refimgs), rf=RF)

    if True:
        for c in topk:
            ref_imgs[c] = [img.detach().cpu() for img in ref_imgs[c]]
        if class_specific_samples:
            hms = {}
            for c in topk:
                hms[c] = fv.get_stats_reference(c, layer_name, [target], mode, (0, n_refimgs), rf=RF,
                                                     heatmap=True, composite=EpsilonPlusFlat(canonizers=canonizers),
                                                     batch_size=32)[target]
        else:
            hms = fv.get_max_reference(topk, layer_name, mode, (0, n_refimgs), heatmap=True,
                                       composite=EpsilonPlusFlat(canonizers=canonizers), rf=RF, batch_size=32)
        masks = get_masks(ref_imgs, hms)
        ref_imgs = get_masked(ref_imgs, hms)

        resize = torchvision.transforms.Resize((224, 224))
        resizem = torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)

        decoder = decoder.float()
        captions = []
        ref_masks = {}

        ann_fname = f"utils/annotations/{dataset_name}/{model_name}.csv"
        os.makedirs(f"utils/annotations/{dataset_name}", exist_ok=True)
        if os.path.isfile(ann_fname):
            annotations = pd.read_csv(ann_fname)
        else:
            annotations = pd.DataFrame({layer_name: [], "nindex": []}).astype({"nindex": int})

        for (samples, masks_, key) in zip(ref_imgs.values(), masks.values(), ref_imgs.keys()):

            masks_ = torch.stack([resizem(torch.tensor(s)[None]) for s in masks_], 0)[None].float()
            ref_masks[key] = masks_[0]
            if key in annotations["nindex"].values:
                if str(annotations[annotations["nindex"] == key][layer_name].values[0]):
                    cap = str(annotations[annotations["nindex"] == key][layer_name].values[0])
                    captions.append(cap)
                else:
                    samples = torch.stack([resize(s) for s in samples], 0)[None].float()
                    outputs = decoder(samples.to(device), masks=masks_.to(device))
                    captions.append(outputs.captions[0])
                    annotations.loc[annotations["nindex"] == key, layer_name] = outputs.captions[0]
            else:
                samples = torch.stack([resize(s) for s in samples], 0)[None].float()
                outputs = decoder(samples.to(device), masks=masks_.to(device))
                captions.append(outputs.captions[0])
                annotations = annotations.append({'nindex': int(key), layer_name: outputs.captions[0]}, ignore_index=True)

        annotations.to_csv(ann_fname, index=False)

    resize = torchvision.transforms.Resize((150, 150))

    fig, axs = plt.subplots(topk_c, 3, gridspec_kw={'width_ratios': [1, 1, n_refimgs / 4]},
                            figsize=(1.6 * 5, 1.6 * topk_c))  # ,

    for r, row_axs in enumerate(axs):
        hm = heatmaps[r]
        ind = topk[r]
        masked_input = np.array(zimage.imgify(mask_img(data[0].detach().cpu(), gauss_p_norm(hm.abs()) > 0.2)))

        grid = make_grid([resize(i) for i in ref_imgs[ind]], nrow=n_refimgs // 2, padding=0)  # topk_img)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        caption = captions[r]
        # gridm = make_grid([resize(i) for i in ref_masks[ind]], nrow=n_refimgs // 2)  # topk_img)
        # gridm = np.array(zimage.imgify(gridm.detach().cpu()))

        for c, ax in enumerate(row_axs):
            if c == 0:
                if r == 0:
                    ax.set_title("heatmap")
                ax.imshow(np.array(zimage.imgify(hm, symmetric=True, cmap="bwr")), cmap="bwr", vmin=0, vmax=255)

                # ax.set_ylabel(f"concept {ind}")
                ax.set_ylabel(f"concept {ind}\n relevance: {(topk_rel[r]*100):2.1f}%")

            elif c == 1:
                if r == 0:
                    ax.set_title("masked input")
                ax.imshow(masked_input)

            elif c == 2:
                if r == 0:
                    ax.set_title("images sharing the concept")
                ax.imshow(grid)
                if '\\n' in caption:
                    c1, c2 = caption.split('\\n')
                    ax.set_ylabel(f"{c1}\n{c2}", rotation=270, labelpad=25, size=9)
                else:
                    ax.set_ylabel(caption, rotation=270, labelpad=15, size=9)
                ax.yaxis.set_label_position("right")
                # ax.contour(gridm.sum(2) > 0, colors="white", linewidths=[0.5])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth('2')

    plt.tight_layout()
    os.makedirs(f"results/crp/{dataset_name}", exist_ok=True)
    os.makedirs(f"results/samples/{dataset_name}", exist_ok=True)
    plt.savefig(f"results/crp/{dataset_name}/sample_{sample}_{model_name}_crp.png", dpi=200)
    zimage.imgify((data[0]).detach().cpu()).save(
        f"results/samples/{dataset_name}/sample_{sample}.png")


def gauss_p_norm(x: Any, sigma: int = 4) -> Any:
    """ Applies Gaussian filter and normalizes"""
    return normalize(gaussian_filter(x, sigma=sigma))


def normalize(a: Any) -> Any:
    """ Applies normalization"""
    if np.abs(a).max() == 0:
        a = a*0 + 1
    return a / np.abs(a).max()


def mask_img(img: torch.Tensor, mask: torch.Tensor, alpha: int = 0.0) -> torch.Tensor:
    """ Masks input sample with mask"""
    minv = 1 - mask  # inverse mask
    return img * mask + img * minv * alpha


def get_masked(imgs: Dict, hms: Dict, thresh=0.2):
    """ Masks img dict from CRP library using heatmaps dict."""
    return {k: [mask_img(img.to(hm), gauss_p_norm(hm) > thresh) for img, hm in zip(imgs[k], hms[k])] for k in
            imgs.keys()}


def get_masks(imgs: Dict, hms: Dict, thresh=0.2):
    """ Masks img dict from CRP library using heatmaps dict."""
    return {k: [gauss_p_norm(hm) > thresh for img, hm in zip(imgs[k], hms[k])] for k in
            imgs.keys()}

if __name__ == "__main__":
    crp_plot()
