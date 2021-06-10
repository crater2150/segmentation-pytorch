import itertools
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import defaultdict


def baseline_to_bbox(baseline, margin_top=12, margin_bot=10, margin_left=10, margin_right=10):
    bounding_box = []
    for ind, point in enumerate(baseline):
        x, y = point
        if ind == 0:
            x = x - margin_left
        if ind == len(baseline) - 1:
            x = x + margin_right
        bounding_box.append((x, y - margin_top))
    for ind, point in enumerate(reversed(baseline)):
        x, y = point
        if ind == len(baseline) - 1:
            x = x - margin_left
        if ind == 0:
            x = x + margin_right
        bounding_box.append((x, y + margin_bot))

    return bounding_box


def crop_image_by_polygon(polygon: List[Tuple[int]], image):
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = image[y:y + h, x:x + w].copy()
    pts = pts - np.amin(pts, axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    return dst, dst2


def get_stroke_width(image):
    skeleton = skeletonize(image)
    skeleton_sum = np.sum(skeleton)
    image_sum = np.sum(image)
    return skeleton_sum / image_sum, image_sum / skeleton_sum


def show_images(img_list, title_list=[], interpolation="nearest"):
    from matplotlib import pyplot as plt
    if len(title_list) < len(img_list):
        # create more titles
        title_list.extend([""] * (len(img_list) - len(title_list)))
    if len(img_list) == 1:
        plt.imshow(img_list[0], interpolation=interpolation)
        plt.title(title_list[0])
        #plt.get_current_fig_manager().window.showMaximized()
        plt.show()
    else:
        f, ax = plt.subplots(1, len(img_list))
        for i, img, title in zip(itertools.count(), img_list, title_list):
            ax[i].imshow(img, interpolation=interpolation)
            ax[i].set_title(title)
        #plt.get_current_fig_manager().window.showMaximized()
        plt.show()


class NewImageReconstructor:
    def __init__(self, labeled_image, total_labels=None, background_color=(0,0,0), undefined_color=(255,255,255)):
        if total_labels is None or total_labels == 0:
            total_labels = int(np.max(labeled_image)) + 1
        self.color_keys = np.tile(np.array(undefined_color, dtype=np.uint8), (total_labels, 1))
        self.labeled_image = labeled_image
        # set label 0 to white
        self.label(0, background_color)

    def label(self, label, color):
        self.color_keys[label, 0] = color[0]
        self.color_keys[label, 1] = color[1]
        self.color_keys[label, 2] = color[2]

    def get_image(self):
        return self.color_keys[self.labeled_image]

    @staticmethod
    def reconstructed_to_binary(reconstructed, background_color=(0, 0, 0)):
        img = np.array(np.where(np.all(reconstructed == background_color, axis=2), 255, 0), dtype=np.uint8)
        assert img.dtype == np.uint8
        return img

    @staticmethod
    def reconstructed_where(reconstructed, positive):
        return np.where(np.all(reconstructed == positive, axis=2), 255, 0, dtype=np.uint8)


class UnionFind:

    @staticmethod
    def from_count(element_count: int):
        return UnionFind(range(element_count))

    @staticmethod
    def from_max_label(max_label: int):
        return UnionFind.from_count(int(max_label) + 1)

    def __init__(self, iterable):  # e.g. use range(100) for 100 nodes
        self.tree = list(iterable)

    def __iter__(self):
        return iter(self.tree)

    def find(self, element: int) -> int:
        tree = self.tree
        cur = element
        if tree[cur] == cur: return cur

        # find parent
        while True:
            cur = tree[cur]
            if tree[cur] == cur: break
        parent = cur

        # compress
        cur = element
        while tree[cur] != cur:
            tree[cur], cur = parent, tree[cur]
        return parent

    def union2(self, element1: int, element2: int):
        par1 = self.find(element1)
        par2 = self.find(element2)
        self.tree[par2] = par1
        return self

    def union_iterable(self, iterable):
        for e1, e2 in iterable:
            self.union2(e1, e2)
        return self

    # returns a dict of lists
    # each item in the dict is a connected component, the list elements
    # describe the indices of all the elements in that CC
    def get_sets(self):
        ccs = defaultdict(list)
        for id, _ in enumerate(self.tree):
            parent = self.find(id)
            ccs[parent].append(id)
        return ccs
