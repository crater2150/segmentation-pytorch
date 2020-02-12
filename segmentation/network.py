from segmentation.dataset import dirs_to_pandaframe, load_image_map_from_file, MaskDataset, compose, post_transforms
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import gc
from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.utils import data
import logging
from segmentation.settings import TrainSettings, PredictorSettings
import segmentation_models_pytorch as sm
from segmentation.dataset import label_to_colors, XMLDataset
from typing import Union
import numpy as np
from pagexml_mask_converter.pagexml_to_mask import MaskGenerator, MaskSetting, BaseMaskGenerator, MaskType, PCGTSVersion

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
console_logger = logging.StreamHandler()
console_logger.setFormatter(logFormatter)
console_logger.terminator = ""
logger.setLevel(logging.DEBUG)
logger.addHandler(console_logger)


class TrainProgressCallback:
    def init(self, total_iters, early_stopping_iters):
        pass

    def update_loss(self, batch: int, loss: float, acc: float):
        pass

    def next_best(self, epoch, acc, n_best):
        pass


class TrainProgressCallbackWrapper:

    def __init__(self,
                 n_iters_per_epoch: int,
                 train_callback: TrainProgressCallback):
        super().__init__()
        self.train_callback = train_callback
        self.n_iters_per_epoch = n_iters_per_epoch
        self.epoch = 0
        self.iter = 0

    def on_batch_end(self, batch, loss, acc, logs=None):
        self.iter = batch + self.epoch * self.n_iters_per_epoch
        self.train_callback.update_loss(self.iter,
                                        loss=loss,
                                        acc=acc)

    def on_epoch_end(self, epoch, acc, wait=0):
        self.epoch = epoch + 1
        self.train_callback.next_best(self.iter, acc, wait)


def pad(tensor, factor=32):
    shape = list(tensor.shape)[2:]
    h_dif = factor - (shape[0] % factor)
    x_dif = factor - (shape[1] % factor)
    x_dif = x_dif if factor != x_dif else 0
    h_dif = h_dif if factor != h_dif else 0
    augmented_image = tensor
    if h_dif != 0 or x_dif != 0:
        augmented_image = torch.nn.functional.pad(input=tensor, pad=[0, x_dif, 0, h_dif])
    return augmented_image


def unpad(tensor, o_shape):
    output = tensor[:, :, :o_shape[0], :o_shape[1]]
    return output


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            shape = list(data.shape)[2:]
            padded = pad(data, 32)

            input = padded.float()

            output = model(input)
            output = unpad(output, shape)
            test_loss += criterion(output, target)
            _, predicted = torch.max(output.data, 1)

            total += target.nelement()
            correct += predicted.eq(target.data).sum().item()
            logger.info('\r Image [{}/{}'.format(idx * len(data), len(test_loader.dataset)))

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Length of Test Set: {} ({:.6f}%)\n'.format(
        test_loss, len(test_loader.dataset),
        100. * correct / total))

    return 100. * correct / total


def train(model, device, train_loader, optimizer, epoch, criterion, accumulation_steps=8, color_map=None,
          callback: TrainProgressCallbackWrapper = None, debug=False):
    def debug_img(mask, target, original, color_map):
        if color_map is not None:
            from matplotlib import pyplot as plt
            mean = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
            mask = torch.argmax(mask, dim=1)
            mask = torch.squeeze(mask)
            original = original.permute(0, 2, 3, 1)
            original = torch.squeeze(original).cpu().numpy()
            original = original * stds
            original = original + mean
            original = original * 255
            original = original.astype(int)
            f, ax = plt.subplots(1, 3, True, True)
            target = torch.squeeze(target)
            ax[0].imshow(label_to_colors(mask=target, colormap=color_map))
            ax[1].imshow(label_to_colors(mask=mask, colormap=color_map))
            ax[2].imshow(original)

            plt.show()

    model.train()
    total_train = 0
    correct_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device, dtype=torch.int64)

        shape = list(data.shape)[2:]
        padded = pad(data, 32)

        input = padded.float()

        output = model(input)
        output = unpad(output, shape)
        loss = criterion(output, target)
        loss = loss / accumulation_steps
        loss.backward()
        _, predicted = torch.max(output.data, 1)
        total_train += target.nelement()
        correct_train += predicted.eq(target.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        logger.info(
            '\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                          len(train_loader.dataset),
                                                                                          100. * batch_idx / len(
                                                                                              train_loader),
                                                                                          loss.item(),
                                                                                          train_accuracy)),
        if (batch_idx + 1) % accumulation_steps == 0:  # Wait for several backward steps
            debug_img(output, target, data, color_map)
            if isinstance(optimizer, Iterable):  # Now we can do an optimizer step
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()
            model.zero_grad()  # Reset gradients tensors
        if callback:
            callback.on_batch_end(batch_idx, loss=loss.item(), acc=train_accuracy)
        gc.collect()


def train_unlabeled(model, device, train_loader, unlabeled_loader,
                    optimizer, epoch, criterion, accumulation_steps=8,
                    color_map=None, train_step=50, alpha_factor=3, epoch_conv=15, debug=False):
    def alpha_weight(epoch):
        return min((epoch / epoch_conv) * alpha_factor, alpha_factor)

    def debug(mask, target, original, color_map):
        if color_map is not None:
            from matplotlib import pyplot as plt
            mean = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
            mask = torch.argmax(mask, dim=1)
            mask = torch.squeeze(mask)
            original = original.permute(0, 2, 3, 1)
            original = torch.squeeze(original).cpu().numpy()
            original = original * stds
            original = original + mean
            original = original * 255
            original = original.astype(int)
            f, ax = plt.subplots(1, 3, True, True)
            target = torch.squeeze(target)
            ax[0].imshow(label_to_colors(mask=target, colormap=color_map))
            ax[1].imshow(label_to_colors(mask=mask, colormap=color_map))
            ax[2].imshow(original)

            plt.show()

    model.train()
    total_train = 0
    correct_train = 0
    for batch_idx, (data, target) in enumerate(unlabeled_loader):
        data = data.to(device)
        shape = list(data.shape)[2:]
        padded = pad(data, 32)

        input = padded.float()
        model.eval()
        with torch.no_grad():
            output_unlabeled = model(input)
            output_unlabeled = unpad(output_unlabeled, shape)
            pseudo_labeled = torch.argmax(output_unlabeled, dim=1)

        model.train()
        output = model(input)
        output = unpad(output, shape)
        if debug:
            debug(output, pseudo_labeled, data, color_map)
        loss = criterion(output, pseudo_labeled)
        loss = (loss * alpha_weight(epoch)) / accumulation_steps
        loss.backward()
        _, predicted = torch.max(output.data, 1)
        total_train += target.nelement()
        correct_train += predicted.eq(pseudo_labeled.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        logger.info(
            '\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                          len(unlabeled_loader.dataset),
                                                                                          100. * batch_idx / len(
                                                                                              unlabeled_loader),
                                                                                          loss.item(),
                                                                                          train_accuracy)),
        if (batch_idx + 1) % accumulation_steps == 0:  # Wait for several backward steps
            # debug(output, target, data, color_map)
            if isinstance(optimizer, Iterable):  # Now we can do an optimizer step
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()
            model.zero_grad()  # Reset gradients tensors
        gc.collect()

        if batch_idx + 1 % train_step == 0:  # used as correction with real data
            print('\n')
            train(model=model, device=device, optimizer=optimizer, train_loader=train_loader,
                  epoch=epoch, criterion=criterion, accumulation_steps=accumulation_steps, color_map=color_map)
    pass


def get_model(architecture, kwargs):
    architecture = architecture.get_architecture()(**kwargs)
    return architecture


class Network(object):

    def __init__(self, settings: Union[TrainSettings, PredictorSettings], color_map=None):
        from segmentation.modules import Architecture

        self.settings = settings
        architecture: Architecture = None
        encoder: str = None
        classes: int = None
        if isinstance(settings, PredictorSettings):
            import os
            with open(str(os.path.splitext(settings.MODEL_PATH)[0]) + '.meta', 'r') as f:
                for x in f.readlines():
                    x = x.strip('\n')
                    if x.startswith('Encoder'):
                        encoder = x.split(" ")[1]
                    if x.startswith('Architecture'):
                        architecture = Architecture(x.split(" ")[1])
                    if x.startswith('Classes'):
                        classes = int(x.split(" ")[1])
            if self.settings.PREDICT_DATASET is not None:
                self.settings.PREDICT_DATASET.preprocessing = sm.encoders.get_preprocessing_fn(encoder)
        elif isinstance(settings, TrainSettings):
            encoder = self.settings.ENCODER
            architecture = self.settings.ARCHITECTURE
            classes = self.settings.CLASSES
            self.settings.TRAIN_DATASET.preprocessing = sm.encoders.get_preprocessing_fn(self.settings.ENCODER)
            self.settings.VAL_DATASET.preprocessing = sm.encoders.get_preprocessing_fn(self.settings.ENCODER)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_params = architecture.get_architecture_params()
        self.model_params['classes'] = classes
        # self.model_params['decoder_use_batchnorm'] = False
        self.model_params['encoder_name'] = encoder
        self.model = get_model(architecture, self.model_params)
        if self.settings.MODEL_PATH:
            try:
                self.model.load_state_dict(torch.load(self.settings.MODEL_PATH, map_location=torch.device(device)))
            except Exception:
                logger.warning('Could not load model weights, ... Skipping\n')

        self.color_map = color_map  # Optional for visualisation of mask data
        self.model.to(self.device)
        self.encoder = encoder

    def train(self, callback=None):

        if not isinstance(self.settings, TrainSettings):
            logger.warning('Settings is of type: {}. Pass settings to network object of type Train to train'.format(
                str(type(self.settings))))
            return

        if callback:
            callback = TrainProgressCallbackWrapper(len(self.settings.TRAIN_DATASET), callback)

        criterion = nn.CrossEntropyLoss()
        self.model.float()
        opt = self.settings.OPTIMIZER.getOptimizer()
        try:
            optimizer1 = opt(self.model.encoder.parameters(), lr=self.settings.LEARNINGRATE_ENCODER)
            optimizer2 = opt(self.model.decoder.parameters(), lr=self.settings.LEARNINGRATE_DECODER)
            optimizer3 = opt(self.model.segmentation_head.parameters(), lr=self.settings.LEARNINGRATE_SEGHEAD)
            optimizer = [optimizer1, optimizer2, optimizer3]
        except:
            optimizer = opt(self.model.parameters(), lr=self.settings.LEARNINGRATE_SEGHEAD)

        train_loader = data.DataLoader(dataset=self.settings.TRAIN_DATASET, batch_size=self.settings.TRAIN_BATCH_SIZE,
                                       shuffle=True, num_workers=self.settings.PROCESSES)
        val_loader = data.DataLoader(dataset=self.settings.VAL_DATASET, batch_size=self.settings.VAL_BATCH_SIZE,
                                     shuffle=False)
        pseudo_loader = None
        if self.settings.PSEUDO_DATASET is not None:
            pseudo_loader = data.DataLoader(dataset=self.settings.PSEUDO_DATASET,
                                            batch_size=self.settings.TRAIN_BATCH_SIZE,
                                            shuffle=True)
        highest_accuracy = 0
        logger.info(str(self.model) + "\n")
        logger.info(str(self.model_params) + "\n")
        logger.info('Training started ...\n"')
        for epoch in range(1, self.settings.EPOCHS):
            if self.settings.PSEUDO_DATASET is not None:
                train_unlabeled(self.model, device=self.device, train_loader=train_loader,
                                unlabeled_loader=pseudo_loader,
                                optimizer=optimizer, epoch=epoch, criterion=criterion,
                                accumulation_steps=self.settings.BATCH_ACCUMULATION,
                                color_map=self.color_map, train_step=50, alpha_factor=3, epoch_conv=15)
            else:
                train(self.model, self.device, train_loader, optimizer, epoch, criterion,
                      accumulation_steps=self.settings.BATCH_ACCUMULATION,
                      color_map=self.color_map,
                      callback=callback)
            accuracy = test(self.model, self.device, val_loader, criterion=criterion)
            if self.settings.OUTPUT_PATH is not None:

                if accuracy > highest_accuracy:
                    logger.info('Saving model to {}\n'.format(self.settings.OUTPUT_PATH + ".torch"))
                    torch.save(self.model.state_dict(), self.settings.OUTPUT_PATH + ".torch")
                    file = self.settings.OUTPUT_PATH + '.meta'
                    with open(file, 'w') as filetowrite:
                        filetowrite.write('Encoder: ' + str(self.settings.ENCODER) + '\n' +
                                          'Architecture: ' + str(self.settings.ARCHITECTURE.value) + '\n' +
                                          'Classes: ' + str(self.settings.CLASSES))

                    highest_accuracy = accuracy
                if callback:
                    callback.on_epoch_end(epoch=epoch, acc=highest_accuracy)

    def predict(self, tta_aug=None, debug=None):
        transforms = tta_aug
        if tta_aug is None:
            import ttach as tta
            transforms = tta.Compose(
                [
                    tta.Scale(scales=[1]),
                ]
            )
        from torch.utils import data

        self.model.eval()

        if not isinstance(self.settings, PredictorSettings):
            logger.warning('Settings is of type: {}. Pass settings to network object of type Train to train'.format(
                str(type(self.settings))))
            return
        predict_loader = data.DataLoader(dataset=self.settings.PREDICT_DATASET,
                                         batch_size=1,
                                         shuffle=False, num_workers=self.settings.PROCESSES)
        with torch.no_grad():
            for idx, (data, target) in enumerate(predict_loader):
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                outputs = []
                o_shape = data.shape
                for transformer in transforms:
                    augmented_image = transformer.augment_image(data)
                    shape = list(augmented_image.shape)[2:]
                    padded = pad(augmented_image, 32)  ## 2**5

                    input = padded.float()
                    output = self.model(input)
                    output = unpad(output, shape)
                    reversed = transformer.deaugment_mask(output)
                    reversed = torch.nn.functional.interpolate(reversed, size=list(o_shape)[2:], mode="nearest")
                    print("original: {} input: {}, padded: {} unpadded {} output {}".format(str(o_shape),
                                                                                            str(shape), str(
                            list(augmented_image.shape)), str(list(output.shape)), str(list(reversed.shape))))
                    outputs.append(reversed)
                stacked = torch.stack(outputs)
                output = torch.mean(stacked, dim=0)
                outputs.append(output)

                def debug(mask, target, original, color_map):
                    if color_map is not None:
                        mean = [0.485, 0.456, 0.406]
                        stds = [0.229, 0.224, 0.225]
                        mask = torch.argmax(mask, dim=1)
                        mask = torch.squeeze(mask)
                        original = original.permute(0, 2, 3, 1)
                        original = torch.squeeze(original).cpu().numpy()
                        original = original * stds
                        original = original + mean
                        original = original * 255
                        original = original.astype(int)
                        extract_baselines(mask, original=original)

                        f, ax = plt.subplots(1, 3, True, True)
                        target = torch.squeeze(target)
                        ax[0].imshow(label_to_colors(mask=target, colormap=color_map))
                        ax[1].imshow(label_to_colors(mask=mask, colormap=color_map))
                        ax[2].imshow(original)
                        ax[2].imshow(label_to_colors(mask=mask, colormap=color_map), cmap='jet', alpha=0.5)

                        plt.show()

                if debug:
                    debug(output, target, data, self.color_map)
                out = output.data.cpu().numpy()
                out = np.transpose(out, (0, 2, 3, 1))
                out = np.squeeze(out)

                def plot(outputs):
                    list_out = []
                    for ind, x in enumerate(outputs):
                        mask = torch.argmax(x, dim=1)
                        mask = torch.squeeze(mask)
                        list_out.append(label_to_colors(mask=mask, colormap=self.color_map))
                    list_out.append(label_to_colors(mask=torch.squeeze(target), colormap=self.color_map))
                    plot_list(list_out)

                if debug:
                    plot(outputs)
                yield out

    def predict_single_image(self, image: np.array, rgb=True, preprocessing=True, tta_aug=None):
        if not isinstance(self.settings, PredictorSettings):
            logger.warning('Settings is of type: {}. Pass settings to network object of type Train to train'.format(
                str(type(self.settings))))
            return
        # from torch.utils import data
        transforms = tta_aug
        if tta_aug is None:
            import ttach as tta
            transforms = tta.Compose(
                [
                    tta.Scale(scales=[1]),
                ]
            )
        from segmentation.util import gray_to_rgb
        self.model.eval()
        preprocessing_fn = sm.encoders.get_preprocessing_fn(self.encoder)
        if rgb:
            image = gray_to_rgb(image)
        result = {"image": image}

        if preprocessing:
            result["image"] = preprocessing_fn(result["image"])
        result = compose([post_transforms()])(**result)
        data = result["image"]
        data = data.unsqueeze(0)

        with torch.no_grad():
            data = data.to(self.device)

            outputs = []
            o_shape = data.shape
            for transformer in transforms:
                augmented_image = transformer.augment_image(data)
                shape = list(augmented_image.shape)[2:]
                padded = pad(augmented_image, 32)  ## 2**5

                input = padded.float()
                output = self.model(input)
                output = unpad(output, shape)
                reversed = transformer.deaugment_mask(output)
                reversed = torch.nn.functional.interpolate(reversed, size=list(o_shape)[2:], mode="nearest")
                print("original: {} input: {}, padded: {} unpadded {} output {}".format(str(o_shape),
                                                                                        str(shape), str(
                        list(augmented_image.shape)), str(list(output.shape)), str(list(reversed.shape))))
                outputs.append(reversed)
            stacked = torch.stack(outputs)
            output = torch.mean(stacked, dim=0)

            out = output.data.cpu().numpy()
            out = np.transpose(out, (0, 2, 3, 1))
            out = np.squeeze(out)

            return out


def extract_baselines(image_map: np.array, base_line_index=1, base_line_border_index=2, original=None):
    # from skimage import measure
    from scipy.ndimage.measurements import label

    base_ind = np.where(image_map == base_line_index)
    base_border_ind = np.where(image_map == base_line_border_index)

    baseline = np.zeros(image_map.shape)
    baseline_border = np.zeros(image_map.shape)
    baseline[base_ind] = 1
    baseline_border[base_border_ind] = 1
    baseline_ccs, n_baseline_ccs = label(baseline, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    baseline_ccs = [np.where(baseline_ccs == x) for x in range(1, n_baseline_ccs + 1)]
    baseline_border_ccs, n_baseline_border_ccs = label(baseline_border, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    baseline_border_ccs = [np.where(baseline_border_ccs == x) for x in range(1, n_baseline_border_ccs + 1)]

    class Cc_with_type(object):
        def __init__(self, cc, type):
            self.cc = cc
            index_min = np.where(cc[1] == min(cc[1]))  # [0]
            index_max = np.where(cc[1] == max(cc[1]))  # [0]

            if type == 'baseline':
                self.cc_left = (np.mean(cc[0][index_min][0]), cc[1][index_min[0]][0])
                self.cc_right = (np.mean(cc[0][index_max][0]), cc[1][index_max[0]][0])

            else:
                self.cc_left = (np.mean(cc[0]), cc[1][index_min[0]][0])
                self.cc_right = (np.mean(cc[0]), cc[1][index_max[0]][0])

            self.type = type

        def __lt__(self, other):
            return self.cc < other

    baseline_ccs = [Cc_with_type(x, 'baseline') for x in baseline_ccs if len(x[0]) > 10]

    baseline_border_ccs = [Cc_with_type(x, 'baseline_border') for x in baseline_border_ccs if len(x[0]) > 10]

    all_ccs = baseline_ccs + baseline_border_ccs
    from segmentation.util import pairwise, angle_between, angle_to

    def calculate_distance_matrix(ccs, length=50):
        distance_matrix = np.zeros((len(ccs), len(ccs)))
        vertical_distance = 10
        for ind1, x in enumerate(ccs):
            for ind2, y in enumerate(ccs):
                if x is y:
                    distance_matrix[ind1, ind2] = 0
                else:
                    distance = 0
                    same_type = 1 if x.type == y.type else 1000

                    def left(x, y):
                        return x.cc_left[1] > y.cc_right[1]

                    def right(x, y):
                        return x.cc_right[1] < y.cc_left[1]

                    ANGLE = 10
                    if left(x, y):
                        angle = angle_to(np.array(y.cc_right), np.array(x.cc_left))
                        distance = x.cc_left[1] - y.cc_right[1]
                        test_angle = ANGLE if distance > 30 else ANGLE * 3 if distance > 5 else ANGLE * 5
                        if test_angle < angle < (360 - test_angle):
                            distance = 99999
                        else:
                            point_c = y.cc_right
                            point_n = x.cc_left

                            x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                            y_points = np.interp(x_points, [point_c[1], point_n[1]], [point_c[0], point_n[0]]).astype(
                                int)
                            indexes = (y_points, x_points)

                            blackness = np.sum(baseline_border[indexes])
                            # print('left' + str(blackness))
                            distance = distance * (blackness * 5000 + 1)

                    elif right(x, y):
                        angle = angle_to(np.array(x.cc_right), np.array(y.cc_left))
                        distance = y.cc_left[1] - x.cc_right[1]
                        test_angle = ANGLE if distance > 30 else ANGLE * 3 if distance > 5 else ANGLE * 5

                        if test_angle < angle < (360 - test_angle):
                            distance = 99999
                        else:
                            distance = y.cc_left[1] - x.cc_right[1]

                            point_c = x.cc_right
                            point_n = y.cc_left

                            x_points = np.arange(start=point_c[1], stop=point_n[1] + 1)
                            y_points = np.interp(x_points, [point_c[1], point_n[1]],
                                                 [point_c[0], point_n[0]]).astype(
                                int)
                            indexes = (y_points, x_points)

                            blackness = np.sum(baseline_border[indexes])
                            distance = distance * (blackness * 5000 + 1)
                    else:
                        distance = 99999

                    distance_matrix[ind1, ind2] = distance * same_type
        return distance_matrix

    matrix = calculate_distance_matrix(all_ccs)

    from sklearn.cluster import DBSCAN
    if np.sum(matrix) == 0:
        print("Empty Image")
        return
    t = DBSCAN(eps=100, min_samples=1, metric="precomputed").fit(matrix)
    debug_image = np.zeros(image_map.shape)
    for ind, x in enumerate(all_ccs):
        debug_image[x.cc] = t.labels_[ind]

    ccs = []
    for x in np.unique(t.labels_):
        ind = np.where(t.labels_ == x)
        line = []
        for d in ind[0]:
            if all_ccs[d].type == 'baseline':
                line.append(all_ccs[d])
        if len(line) > 0:
            ccs.append((np.concatenate([x.cc[0] for x in line]), np.concatenate([x.cc[1] for x in line])))

    ccs = [list(zip(x[0], x[1])) for x in ccs]

    from itertools import chain
    from typing import List, Tuple
    from collections import defaultdict
    def normalize_connected_components(cc_list: List[List[Tuple[int, int]]]):
        # Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
        def normalize(point_list):
            normalized_cc_list = []
            for cc in point_list:
                cc_dict = defaultdict(list)
                for y, x in cc:
                    cc_dict[x].append(y)
                normalized_cc = []
                # for key, value in cc_dict.items():
                for key in sorted(cc_dict.keys()):
                    value = cc_dict[key]
                    normalized_cc.append([int(np.floor(np.mean(value) + 0.5)), key])
                normalized_cc_list.append(normalized_cc)
            return normalized_cc_list

        return normalize(cc_list)

    ccs = normalize_connected_components(ccs)

    plt.imshow(original)
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.uint8(original))
    draw = ImageDraw.Draw(im)

    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255)]

    for ind, x in enumerate(ccs):
        t = list(chain.from_iterable(x))
        # print(t)
        a = t[::-1]

        # t = list(zip(*x))
        # print(t)
        draw.line(a, fill=colors[ind % len(colors)], width=2)
        pass
    im.show()
    from matplotlib import pyplot
    import cycler
    n = 15
    color = pyplot.cm.rainbow(np.linspace(0, 1, n))
    debug_image[debug_image == 0] = 255
    plt.imshow(debug_image, cmap="gist_ncar")
    plt.show()


def plot_list(lsit):
    import matplotlib.pyplot as plt
    import numpy as np
    # print(len(lsit))
    images_per_row = 4
    rows = int(np.ceil(len(lsit) / images_per_row))
    f, ax = plt.subplots(rows, images_per_row, True, True)
    ind = 0
    row = 0
    for x in lsit:
        if rows > 1:
            ax[row, ind].imshow(x)
        else:
            ax[ind].imshow(x)
        ind += 1
        if ind == images_per_row:
            row += 1
            ind = 0

    plt.show()

    def show_images(images, cols=1, titles=None):
        """Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert ((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    # show_images(images=lsit, cols=1)


if __name__ == '__main__':
    '''
    'https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb'
    a = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/train/images/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/train/masks/'])

    b = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/test/images/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/test/masks/']
    )
    b = b[:20]
    map = load_image_map_from_file(
        '/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/image_map.json')
    dt = MaskDataset(a, map, preprocessing=None, transform=compose([post_transforms()]))
    d_test = MaskDataset(b, map, preprocessing=None, transform=compose([post_transforms()]))
    '''
    a = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/train/image/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/train/page/'])

    b = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/test/image/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/test/page/'])

    c = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/HBR2013/images/'],
        ['/home/alexander/Dokumente/HBR2013/masks/']
    )
    d = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/test/image/'],
        ['/home/alexander/Dokumente/dataset/test/mask/']
    )
    e = dirs_to_pandaframe(
        ['/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/OCR-D/images/'],
        ['/mnt/sshfs/scratch/Datensets_Bildverarbeitung/page_segmentation/OCR-D/images/']
    )
    f = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/diva-his/all-privateTest/img-CB55-privateTest/CB55/'],
        ['/home/alexander/Dokumente/dataset/diva-his/all-privateTest/img-CB55-privateTest/CB55/']
    )
    g = dirs_to_pandaframe(
        ['/mnt/sshfs/hartelt/datasets/Tristrant_1484/images/',
         '/mnt/sshfs/hartelt/datasets/Narrenschiff_1553/images/',
         '/mnt/sshfs/hartelt/datasets/Narrenschiff_1512/images/',
         '/mnt/sshfs/hartelt/datasets/Narrenschiff_1509/images/',
         '/mnt/sshfs/hartelt/datasets/Melusina_1474/images/',
         '/mnt/sshfs/hartelt/datasets/La_grant_nef_des_folz_du_monde_1499/images/',
         '/mnt/sshfs/hartelt/datasets/Franck_Chronica_1536_2/images/',
         '/mnt/sshfs/hartelt/datasets/Franck_Chronica_1536_1/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Tusculanen_1538/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Querela_Martini_Luteri_1555/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Oratio_senatoria_de_bello_Turcico_1542/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Oratio_in_declaratione_magistrorum_1563/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Melanchthonvita_1566/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Historiae_Iesu_1566/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Epistulae_familiares_1595/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius-Epistulae_Eobani-1561/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Epistulae_1583/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Epistolae_familiares_1583/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Elementa_rhetoricae_1541/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_De_Helio_Eobano_Hesso_1553/images/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Decuriae_1594/images/',
         ],
        ['/mnt/sshfs/hartelt/datasets/Tristrant_1484/masks/',
         '/mnt/sshfs/hartelt/datasets/Narrenschiff_1553/masks/',
         '/mnt/sshfs/hartelt/datasets/Narrenschiff_1512/masks/',
         '/mnt/sshfs/hartelt/datasets/Narrenschiff_1509/masks/',
         '/mnt/sshfs/hartelt/datasets/Melusina_1474/masks/',
         '/mnt/sshfs/hartelt/datasets/La_grant_nef_des_folz_du_monde_1499/masks/',
         '/mnt/sshfs/hartelt/datasets/Franck_Chronica_1536_2/masks/',
         '/mnt/sshfs/hartelt/datasets/Franck_Chronica_1536_1/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Tusculanen_1538/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Querela_Martini_Luteri_1555/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Oratio_senatoria_de_bello_Turcico_1542/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Oratio_in_declaratione_magistrorum_1563/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Melanchthonvita_1566/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Historiae_Iesu_1566/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Epistulae_familiares_1595/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius-Epistulae_Eobani-1561/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Epistulae_1583/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Epistolae_familiares_1583/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Elementa_rhetoricae_1541/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_De_Helio_Eobano_Hesso_1553/masks/',
         '/mnt/sshfs/hartelt/datasets/Camerarius_Decuriae_1594/masks/',
         ]
    )
    map = load_image_map_from_file(
        '/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/image_map.json')
    from segmentation.dataset import base_line_transform

    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    dt = XMLDataset(a[:5], map, transform=compose([base_line_transform()]),
                    mask_generator=MaskGenerator(settings=settings))
    d_test = XMLDataset(b[:5], map, transform=compose([base_line_transform()]),
                        mask_generator=MaskGenerator(settings=settings))
    d_predict = MaskDataset(e[:5], map,
                            transform=compose([base_line_transform()]))  # transform=compose([base_line_transform()]))
    from segmentation.settings import TrainSettings

    setting = TrainSettings(CLASSES=len(map), TRAIN_DATASET=dt, VAL_DATASET=d_test, ENCODER='resnet18',
                            OUTPUT_PATH="/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/model_test_meta_1")
    # MODEL_PATH='/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/model_hist_cbad.torch')
    p_setting = PredictorSettings(PREDICT_DATASET=d_predict,
                                  MODEL_PATH='/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/model9090.torch')
    trainer = Network(p_setting, color_map=map)
    # trainer.train()
    from PIL import Image

    a = np.array(Image.open(a.get('images')[0]))
    data = trainer.predict_single_image(a)
    for x in trainer.predict():
        print(x.shape)
