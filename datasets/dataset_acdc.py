import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)

    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = ndimage.zoom(
                image,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=3
            )
            label = ndimage.zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0
            )

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {
            "image": image,
            "label": label,
            "case_name": sample.get("case_name", "")
        }

        return sample


class ACDC_dataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform

        self.sample_list = sorted([
            f for f in os.listdir(base_dir)
            if f.endswith(".npz")
        ])

        print(f"ACDC {split} set path: {base_dir}")
        print(f"The length of ACDC {split} set is: {len(self.sample_list)}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        filepath = os.path.join(self.base_dir, case_name)

        data = np.load(filepath)

        if "img" in data.files:
            image = data["img"]
        else:
            image = data["image"]

        label = data["label"]

        image = image.astype(np.float32)
        label = label.astype(np.int64)

        sample = {
            "image": image,
            "label": label,
            "case_name": case_name.replace(".npz", "")
        }

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample["image"] = torch.from_numpy(image.astype(np.float32))
            sample["label"] = torch.from_numpy(label.astype(np.float32))

        return sample
