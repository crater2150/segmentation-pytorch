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
            # test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target)
            _, predicted = torch.max(output.data, 1)

            total += target.nelement()
            correct += predicted.eq(target.data).sum().item()
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            logger.info('\r Image [{}/{}'.format(idx * len(data), len(test_loader.dataset)))

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / total / len(test_loader.dataset)))
    return 100. * correct / total / len(test_loader.dataset)


def train(model, device, train_loader, optimizer, epoch, criterion, accumulation_steps=8, color_map=None):
    def debug(mask, target, original, color_map):
        if color_map is not None:
            from matplotlib import pyplot as plt
            mean = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
            mask = torch.argmax(mask, dim=1)
            mask = torch.squeeze(mask)
            # print(original.shape)
            original = original.permute(0, 2, 3, 1)
            # print(original.shape)
            original = torch.squeeze(original).cpu().numpy()
            # print(sm.encoders.get_preprocessing_params("resnet34"))
            # print(get_preprocessing_params(encoder_name, pretrained=pretrained))
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
        # sys.stdout.flush()
        # , end="",
        #    flush=True)
        if (batch_idx + 1) % accumulation_steps == 0:  # Wait for several backward steps
            debug(output, target, data, color_map)
            if isinstance(optimizer, Iterable):  # Now we can do an optimizer step
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()
            model.zero_grad()  # Reset gradients tensors
        gc.collect()


def get_model(architecture, kwargs):
    architecture = architecture.get_architecture()(**kwargs)
    return architecture


class Network(object):

    def __init__(self, settings: Union[TrainSettings, PredictorSettings], color_map=None):
        self.settings = settings

        if isinstance(settings, PredictorSettings):
            self.settings.PREDICT_DATASET.preprocessing = sm.encoders.get_preprocessing_fn(self.settings.ENCODER)
        elif isinstance(settings, TrainSettings):
            self.settings.TRAIN_DATASET.preprocessing = sm.encoders.get_preprocessing_fn(self.settings.ENCODER)
            self.settings.VAL_DATASET.preprocessing = sm.encoders.get_preprocessing_fn(self.settings.ENCODER)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_params = self.settings.ARCHITECTURE.get_architecture_params()
        self.model_params['classes'] = self.settings.CLASSES
        self.model_params['decoder_use_batchnorm'] = False
        self.model_params['encoder_name'] = self.settings.ENCODER
        self.model = get_model(self.settings.ARCHITECTURE, self.model_params)
        if self.settings.MODEL_PATH:
            try:
                self.model.load_state_dict(torch.load(self.settings.MODEL_PATH, map_location=torch.device(device)))
            except Exception:
                logger.warning('Could not load model weights, ... Skipping\n')

        self.color_map = color_map  # Optional for visualisation of mask data
        self.

    def train(self):

        if not isinstance(self.settings, TrainSettings):
            logger.warning('Settings is of type: {}. Pass settings to network object of type Train to train'.format(
                str(type(self.settings))))
            return

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
        highest_accuracy = 0
        logger.info(str(self.model) + "\n")
        logger.info(str(self.model_params) + "\n")
        logger.info('Training started ...\n"')
        for epoch in range(1, self.settings.EPOCHS):
            train(self.model, self.device, train_loader, optimizer, epoch, criterion,
                  accumulation_steps=self.settings.BATCH_ACCUMULATION,
                  color_map=self.color_map)
            accuracy = test(self.model, self.device, val_loader, criterion=criterion)
            if self.settings.OUTPUT_PATH is not None:
                if accuracy > highest_accuracy:
                    logger.info('Saving model to {}\n'.format(self.settings.OUTPUT_PATH))
                    torch.save(self.model.state_dict(), self.settings.OUTPUT_PATH)
                    highest_accuracy = accuracy

    def predict(self):

        from torch.utils import data

        self.model.eval()

        if not isinstance(self.settings, PredictorSettings):
            logger.warning('Settings is of type: {}. Pass settings to network object of type Train to train'.format(
                str(type(self.settings))))
            return
        predict_loader = data.DataLoader(dataset=self.settings.PREDICT_DATASET,
                                         batch_size=1,
                                         shuffle=False, num_workers=self.settings.PROCESSES)
        total = 0

        import ttach as tta
        # tta_model = tta.SegmentationTTAWrapper(self.model, tta.aliases.multiscale_transform(scales=[1.2]), merge_mode='mean')
        transforms = tta.Compose(
            [
                # tta.HorizontalFlip(),
                tta.Scale(scales=[1]),
            ]
        )
        with torch.no_grad():
            for idx, (data, target) in enumerate(predict_loader):
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                outputs = []
                o_shape = data.shape
                for transformer in transforms:
                    augmented_image = transformer.augment_image(data)
                    shape = list(augmented_image.shape)[2:]
                    padded = pad(augmented_image, 32)

                    input = padded.float()
                    # print(input.shape)
                    output = self.model(input)
                    output = unpad(output, shape)
                    reversed = transformer.deaugment_mask(output)
                    reversed = torch.nn.functional.interpolate(reversed, size=list(o_shape)[2:], mode="nearest")
                    print("original: {} input: {}, padded: {} unpadded {} output {}".format(str(o_shape),
                                                                                            str(shape), str(
                            list(augmented_image.shape)), str(list(output.shape)), str(list(reversed.shape))))
                    outputs.append(reversed)

                # mean(outputs)
                # plt.show()

                stacked = torch.stack(outputs)
                output = torch.mean(stacked, dim=0)
                outputs.append(output)

                def debug(mask, target, original, color_map):
                    if color_map is not None:
                        mean = [0.485, 0.456, 0.406]
                        stds = [0.229, 0.224, 0.225]
                        mask = torch.argmax(mask, dim=1)
                        mask = torch.squeeze(mask)
                        # print(original.shape)
                        original = original.permute(0, 2, 3, 1)
                        # print(original.shape)
                        original = torch.squeeze(original).cpu().numpy()
                        # print(sm.encoders.get_preprocessing_params("resnet34"))
                        # print(get_preprocessing_params(encoder_name, pretrained=pretrained))
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

                debug(output, target, data, self.color_map)

                out = output.data.cpu().numpy()
                out = np.transpose(out, (0, 2, 3, 1))
                out = np.squeeze(out)

                def plot(outputs):
                    # plt.imshow(label_to_colors(mask=torch.squeeze(target), colormap=self.color_map))
                    # plt.show()
                    # plt.figure()
                    # f, ax = plt.subplots(1, 4, True, True)
                    list_out = []
                    for ind, x in enumerate(outputs):
                        mask = torch.argmax(x, dim=1)
                        mask = torch.squeeze(mask)
                        # ax[ind] =
                        list_out.append(label_to_colors(mask=mask, colormap=self.color_map))

                        # plt.imshow(label_to_colors(mask=mask, colormap=self.color_map))
                        # plt.show()
                    list_out.append(label_to_colors(mask=torch.squeeze(target), colormap=self.color_map))
                    # plt.show()
                    plot_list(list_out)

                plot(outputs)
                yield out

    def predict_single_image(self):
        pass


def plot_list(lsit):
    import matplotlib.pyplot as plt
    import numpy as np

    # f, ax = plt.subplots(1, len(lsit), True, True)
    # for ind, x in enumerate(lsit):
    #    ax[ind].imshow(x)
    # plt.show()
    import matplotlib.pyplot as plt
    import numpy as np
    print(len(lsit))
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
    print(c)
    map = load_image_map_from_file(
        '/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/image_map.json')
    from segmentation.dataset import base_line_transform

    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    dt = XMLDataset(a, map, transform=compose([base_line_transform()]), mask_generator=MaskGenerator(settings=settings))
    d_test = XMLDataset(b, map, transform=compose([base_line_transform()]),
                        mask_generator=MaskGenerator(settings=settings))
    d_predict = MaskDataset(c, map, transform=compose([base_line_transform()]))
    from segmentation.settings import TrainSettings

    setting = TrainSettings(CLASSES=len(map), TRAIN_DATASET=dt, VAL_DATASET=d_test, OUTPUT_PATH="model.torch",
                            MODEL_PATH='/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/model.torch')
    p_setting = PredictorSettings(CLASSES=len(map), PREDICT_DATASET=d_predict,
                                  MODEL_PATH='/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/model.torch')
    trainer = Network(setting, color_map=map)
    # for x in trainer.predict():
    #    print(x.shape)
    trainer.train()
