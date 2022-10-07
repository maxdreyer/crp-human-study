'''
The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2021 Fraunhofer-Gesellschaft zur Förderung der
angewandten Forschung e.V. (Fraunhofer)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted for purpose of testing the functionalities of
this software provided that the following conditions are met:
*     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
*     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
*     Neither the names of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
'''
import os

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def imagenet_corrupted_train(preprocessing=True):
    if preprocessing:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return ImageNetDataset("/hardd/ImageNet-complete", train=True, transform=transform)   # change to imagenet directory


def imagenet_corrupted_test(preprocessing=True):
    if preprocessing:
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    return ImageNetDataset("/hardd/ImageNet-complete", train=False, transform=transform)  # change to imagenet directory


class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, root, *args, validate=False, train=True, use_precomputed_labels=False,
                 labels_path=None, transform=None, **kwargs):
        """ImageNet root folder is expected to have two directories: train and val."""

        if train and validate == train:
            raise ValueError('Train and validate can not be True at the same time.')
        if use_precomputed_labels and labels_path is None:
            raise ValueError('If use_precomputed_labels=True the labels_path is necessary.')

        if train:
            root = os.path.join(root, 'train')
        elif validate:
            root = os.path.join(root, 'val_mpeg')
        else:
            root = os.path.join(root, 'val')

        super().__init__(root, transform=transform, *args, **kwargs)
        self.preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if validate and use_precomputed_labels:
            df = pd.read_csv(labels_path, sep='\t')
            df.input_path = df.input_path.apply(lambda x: os.path.join(root, x))
            mapping = dict(zip(df.input_path, df.pred_class))
            # self.samples = [(mapping[x[0]], x[1]) for x in self.samples]
            self.samples = [(x[0], mapping[x[0]]) for x in self.samples]
            self.targets = [x[1] for x in self.samples]


        self.CLASSES = [976, 719, 263, 981, 486, 429, 897, 150, 597, 96, 718, 75, 247, 398, 354, 665, 877, 299, 567,
               686, 308, 575, 37, 386, 705, 746, 928, 245, 546, 518]

        self.padding = 10
        self.transform_corrupted = torchvision.transforms.Compose([
            torchvision.transforms.Pad(self.padding, fill=0),
            torchvision.transforms.Resize(224)
        ])

    def __getitem__(self, item):
        img, target = super().__getitem__(item)
        if target in self.CLASSES:
            img = self.transform_corrupted(img)
        return img, target

    @staticmethod
    def reverse_augmentation(data: torch.Tensor) -> torch.Tensor:
        data = data.clone() + 0
        mean = torch.Tensor((0.485, 0.456, 0.406)).to(data)
        var = torch.Tensor((0.229, 0.224, 0.225)).to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255)


if __name__ == "__main__":
    ds = imagenet_corrupted_train()
    loader = DataLoader(ds, batch_size=50, num_workers=50)
    for i, (data, target) in enumerate(iter(loader)):
        print(i)
