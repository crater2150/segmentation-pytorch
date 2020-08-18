from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


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
