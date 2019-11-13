from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import json
from ast import literal_eval


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


if __name__ == '__main__':
    a = dirs_to_pandaframe('/mnt/sshfs/hartelt/datasets/all/images/', '/mnt/sshfs/hartelt/datasets/all/masks/')
    print(a.get('masks'))
    print(a.index.tolist())
    map = load_image_map_from_file('/mnt/sshfs/hartelt/datasets/all/image_map.json')
    dt = MaskDataset(a, map, 'train')
    i, m = dt.__getitem__(77)
    print(i)
    print(m)
    pass