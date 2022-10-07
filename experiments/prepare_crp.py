import click
import torch.cuda
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from zennit.torchvision import ResNetCanonizer

from datasets_ import get_dataset
import torchvision.transforms as T

from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat

from models import get_model


@click.command()
@click.option("--model_name", default="vgg16")
@click.option("--dataset_name", default="imagenet")
@click.option("--batch_size", default=64)
@click.option("--ckpt", default=None)
def prepare_crp(model_name, dataset_name, batch_size, ckpt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name)
    if ckpt:
        model.load_state_dict(torch.load(ckpt))
    model = model.to(device)
    model.eval()

    canonizers = [ResNetCanonizer()]
    composite = EpsilonPlusFlat(canonizers)
    dataset = get_dataset(dataset_name)["test"](preprocessing=False)

    cc = ChannelConcept()

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer: cc for layer in layer_names}
    print(layer_names)
    attribution = CondAttribution(model)

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                              path=f"{model_name}_{dataset_name}")

    fv.run(composite, 0, int(len(dataset)), batch_size, 100)


if __name__ == "__main__":
    prepare_crp()