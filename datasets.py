# from __future__ import division

import os
import argparse
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# from pycocotools.coco import COCO
import torch.optim as optim
import random
import warnings
from PIL import Image
from PIL import ImageFile
import numpy as np

from augmentations import AUGMENTATION_TRANSFORMS

ImageFile.LOAD_TRUNCATED_IMAGES = True

# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from utils import worker_seed_set

# from terminaltables import AsciiTable
# from torchsummary import summary

class MyDataset(Dataset):
    def __init__(self, data_root, transform=None, target_transform=None):
        self.root      = os.path.join(data_root, 'images/trainval35k')
        self.coco      = COCO(os.path.join(data_root, 'annotations/instances_trainval35k.json'))
        self.ids       = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        # print("height / width = {}/{}".format(height, width))

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, label = self.transform(img, target[:, :4], target[:, 4])

            # to RGB
            img = img[:, :, (2, 1, 0)]

            # hstack : 配列を結合する
            target = np.hstack((boxes, np.expand_dims(labels, axis-1)))

        return torch.from_numpy(img).permute(2, 0, 1), target

    def __len__(self):
        return len(self.ids)

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

""" Dataset (from eriklindernoren) """
class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        # print("bb_targets.shape :", bb_targets.shape)
        # print("label_path :", label_path)
        # print("bb_tagets :", bb_targets)
        # print(np.shape(boxes))
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        # print("batch :", np.shape(list(zip(*batch))))
        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

""" dataloader (from eriklindernoren) """
def _create_data_loader(img_path, batch_size, img_size, n_cpu=8, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
        # )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader
