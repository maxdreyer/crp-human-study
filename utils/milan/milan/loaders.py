"""Tools for loading pretrained MILAN models."""
from typing import Any

import utils.milan.milannotations.loaders
from utils.milan.milan import decoders
from utils.milan.utils import hubs


def hub() -> hubs.ModelHub:
    """Create the model hub."""
    configs = {}
    for group in utils.milan.milannotations.loaders.DATASET_GROUPINGS:
        if group.startswith('NOT_'):
            continue
        for rerank_with_clip in (False, True):
            key = f'{group}+clip' if rerank_with_clip else group
            configs[key] = hubs.ModelConfig(
                decoders.DecoderWithCLIP.load
                if rerank_with_clip else decoders.Decoder.load,
                url=f'{hubs.HOST}/models/milan-{group.replace("/", "_")}.pth',
                requires_path=True,
                load_weights=False,
                map_location='cpu',
            )
    return hubs.ModelHub(**configs)


def pretrained(config: str = 'base', **kwargs: Any) -> decoders.Decoder:
    """Return a pretrained MILAN model."""
    model = hub().load(config, **kwargs)
    assert isinstance(model, decoders.Decoder), model
    return model
