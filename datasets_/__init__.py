from typing import Dict, Any

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

from datasets_.imagenet import imagenet_train, imagenet_test
from datasets_.imagenet_corrupted import imagenet_corrupted_train, imagenet_corrupted_test

DATASETS = {
    "imagenet":
        {"train": imagenet_train,
         "test": imagenet_test,
         "n_classes": 1000},
    "imagenet_corrupted":
        {"train": imagenet_corrupted_train,
         "test": imagenet_corrupted_test,
         "n_classes": 1000},
}


def get_dataset(dataset_name: str) -> Dict[str, Any]:
    print("INIT", dataset_name)
    try:
        dataset = DATASETS[dataset_name]
        return dataset
    except KeyError:
        print(f"DATASET {dataset_name} not defined.")
        exit()


def get_sample(dataset: Dataset, sample_id: int, device):
    # get sample and push it to device
    data = dataset[sample_id]
    processed = []
    for x in data:
        # print(type(x))
        if isinstance(x, torch.Tensor) and len(x.shape):
            processed.append(x[None, :].to(device))
        elif isinstance(x, int) or isinstance(x, np.int) or isinstance(x, torch.Tensor) or isinstance(x, np.int64):
            processed.append(torch.Tensor([x])[None, :].to(device))
        else:
            print(f"data sample of type {type(x)} not put to device.")
            if "labels" in x:
                processed.append(torch.Tensor(x['labels']))
            else:
                print(x)
    return processed
