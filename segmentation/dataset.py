from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch.transforms import ToTensorV2
import json
from ast import literal_eval
import random
from matplotlib import pyplot as plt

from skimage.morphology import remove_small_holes
import albumentations as albu

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def color_to_label(mask, colormap: dict):
    out = np.zeros(mask.shape[0:2], dtype=np.int32)

    if mask.ndim == 2:
        return mask.astype(np.int32) / 255
    if mask.shape[2] == 2:
        return mask[:, :, 0].astype(np.int32) / 255
    mask = mask.astype(np.uint32)
    mask = 256 * 256 * mask[:, :, 0] + 256 * mask[:, :, 1] + mask[:, :, 2]
    for color, label in colormap.items():
        color_1d = 256 * 256 * color[0] + 256 * color[1] + color[2]
        out += (mask == color_1d) * label[0]
    return out


def label_to_colors(mask, colormap: dict):
    out = np.zeros(mask.shape + (3,), dtype=np.int64)
    for color, label in colormap.items():
        trues = np.stack([(mask == label[0])] * 3, axis=-1)
        out += np.tile(color, mask.shape + (1,)) * trues

    out = np.ndarray.astype(out, dtype=np.uint8)
    return out


class MaskDataset(Dataset):
    def __init__(self, df, color_map, phase):
        self.df = df
        self.color_map = color_map
        self.augmentation = get_transforms(phase)
        self.index = self.df.index.tolist()

    def __getitem__(self, item):
        image_id, mask_id = self.df.get('images')[item], self.df.get('masks')[item]
        image = np.asarray(Image.open(image_id))
        mask = np.asarray(Image.open(mask_id))

        if mask.ndim == 3:
            mask = color_to_label(mask, self.color_map)
        elif mask.ndim == 2:
            u_values = np.unique(mask)
            for ind, x in enumerate(u_values):
                mask[mask == x] = ind
        return image, to_categorical(mask, len(self.color_map))

    def __len__(self):
        return len(self.index)


def get_transforms(phase):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    list_transforms.extend(
        [
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def listdir(dir, postfix="", not_postfix=False):
    if dir is None:
        return None
    if len(postfix) > 0 and not_postfix:
        return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if not f.endswith(postfix)]
    else:
        return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if f.endswith(postfix)]


def dirs_to_pandaframe(images_dir, masks_dir, verify_filenames: bool = True):
    img, m = listdir(images_dir), listdir(masks_dir)
    if verify_filenames:
        def filenames(fn, postfix=None):
            if postfix and len(postfix) > 0:
                fn = [f[:-len(postfix)] if f.endswith(postfix) else f for f in fn]

            x = {os.path.basename(f).split('.')[0]: f for f in fn}
            return x
        img_dir = filenames(img)
        mask_dir = filenames(m)
        base_names = set(img_dir.keys()).intersection(set(mask_dir.keys()))

        img = [img_dir.get(basename) for basename in base_names]
        m = [mask_dir.get(basename) for basename in base_names]

    else:
        base_names = None

    df = pd.DataFrame(data={'images': img, 'masks': m})

    return df


def load_image_map_from_file(path):
    if not os.path.exists(path):
        raise Exception("Cannot open {}".format(path))

    with open(path) as f:
        data = json.load(f)
    color_map = {literal_eval(k): v for k, v in data.items()}
    return color_map


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        #albu.RandomRotate90(),
        albu.CoarseDropout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3, border_mode=0, value=255, mask_value=[255,255,255]),
    ]

    return result


def resize_transforms(image_size=1280):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(image_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size * 2, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
            #random_crop_big
        ], p=1)
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    # convenient if ypu want to add extra targets, e.g. binary input
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def show_examples(name: str, image: np.ndarray, binary: np.ndarray, mask: np.ndarray):
    foreground = np.stack([(binary)] * 3, axis=-1)
    inv_binary = 1 - binary
    inv_binary = np.stack([inv_binary] * 3, axis=-1)
    overlay_mask = mask.copy()
    overlay_mask[foreground == 0] = 0
    inverted_overlay_mask = mask.copy()
    inverted_overlay_mask[inv_binary == 0] = 0

    plt.figure(figsize=(10, 14))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")

    plt.subplot(1, 3, 3)
    plt.imshow(inverted_overlay_mask)
    plt.title(f"Binary: {name}")



def show(index: int, image, mask, transforms=None) -> None:

    image = Image.open(image)
    image = np.asarray(image)
    print(image.shape)

    mask = np.array(Image.open(mask))

    print(mask.shape)
    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp['image']
        mask = temp['mask']
    bin_og = Image.fromarray(image)
    bin_og = bin_og.convert('1')
    binary = np.array(bin_og)
    binary = np.asarray(binary)
    binary = remove_small_holes(binary, 3, True)
    show_examples(index, image, binary, mask)


def show_random(df, transforms=None) -> None:
    length = len(df)
    index = random.randint(0, length - 1)
    image = df.get('images')[index]
    mask = df.get('masks')[index]
    print(image)
    print(mask)
    show(index, image, mask, transforms)
    plt.show()


if __name__ == '__main__':
    'https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb'
    a = dirs_to_pandaframe('/mnt/sshfs/hartelt/datasets/all/images/', '/mnt/sshfs/hartelt/datasets/all/masks/')
    map = load_image_map_from_file('/mnt/sshfs/hartelt/datasets/all/image_map.json')
    dt = MaskDataset(a, map, 'train')
    i, m = dt.__getitem__(77)
    train_transforms = compose([
    resize_transforms(),
    hard_transforms(),
    post_transforms()
    ])
    valid_transforms = compose([pre_transforms(), post_transforms()])

    show_transforms = compose([resize_transforms(), hard_transforms()])
    while True:
       show_random(a, transforms=show_transforms)

    pass

