from segmentation.dataset import dirs_to_pandaframe, load_image_map_from_file, MaskDataset, compose, post_transforms
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import gc
from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.utils import data
import logging
from segmentation.settings import TrainSettings

logger = logging.getLogger('Logger')


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data.float())
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def train(model, device, train_loader, optimizer, epoch, criterion, accumulation_steps=8, color_map=None):
    def debug(mask, color_map):
        if color_map is not None:
            from matplotlib import pyplot as plt
            from segmentation.dataset import label_to_colors
            mask = torch.argmax(mask, dim=1)
            mask = torch.squeeze(mask)
            plt.imshow(label_to_colors(mask=mask, colormap=color_map))
            plt.show()
    model.train()
    total_train = 0
    correct_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        output = model(data.float())
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
                                                                                          train_accuracy), end="",
            flush=True)
        if (batch_idx + 1) % accumulation_steps == 0:  # Wait for several backward steps
            debug(output, color_map)
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




class Trainer(object):

    def __init__(self, settings: TrainSettings, color_map=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.settings = settings
        self.model_params = self.settings.ARCHITECTURE.get_architecture_params()
        self.model_params['classes'] = self.settings.CLASSES
        self.model = get_model(self.settings.ARCHITECTURE, self.model_params)
        if self.settings.MODEL_PATH:
            try:
                self.model.load_state_dict(torch.load(self.settings.MODEL_PATH))
            except Exception:
                logger.warning('Could not load model weights, ... Skipping')

        self.color_map = color_map  # Opt ional for visualisation of mask data

    def train(self):

        criterion = nn.CrossEntropyLoss()
        self.model.float()
        opt = self.settings.OPTIMIZER.getOptimizer()
        try:
            optimizer1 = opt(self.model.encoder.parameters(), lr=1e-3)
            optimizer2 = opt(self.model.decoder.parameters(), lr=1e-3)
            optimizer3 = opt(self.model.segmentation_head.parameters(), lr=1e-3)
            optimizer = [optimizer1, optimizer2, optimizer3]
        except:
            optimizer = opt(self.model.parameters(), lr=1e-3)

        train_loader = data.DataLoader(dataset=self.settings.TRAIN_DATASET, batch_size=self.settings.TRAIN_BATCH_SIZE,
                                       shuffle=True, num_workers=self.settings.PROCESSES)
        val_loader = data.DataLoader(dataset=self.settings.VAL_DATASET, batch_size=self.settings.VAL_BATCH_SIZE, shuffle=False)
        highest_accuracy = 0
        for epoch in range(1, self.settings.EPOCHS):
            logger.info('Training started ...')
            logger.info(self.model)
            logger.info(str(self.model_params))
            train(self.model, self.device, train_loader, optimizer, epoch, criterion,
                  accumulation_steps=self.settings.BATCH_ACCUMULATION,
                  color_map=self.color_map)
            accuracy = test(self.model, self.device, val_loader, criterion=criterion)
            if self.settings.OUTPUT_PATH is not None:
                if accuracy > highest_accuracy:
                    torch.save(self.model.state_dict(), self.settings.OUTPUT_PATH)
                    highest_accuracy = accuracy


if __name__ == '__main__':
    'https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb'
    a = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/train/images/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/train/masks/'])

    b = dirs_to_pandaframe(
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/test/images/'],
        ['/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/test/masks/']
    )
    map = load_image_map_from_file(
        '/home/alexander/Dokumente/dataset/READ-ICDAR2019-cBAD-dataset/dataset-test/image_map.json')
    dt = MaskDataset(a, map, preprocessing=None, transform=compose([post_transforms()]))
    d_val = MaskDataset(b, map, preprocessing=None, transform=compose([post_transforms()]))

    from segmentation.settings import TrainSettings
    setting = TrainSettings(CLASSES=len(map), TRAIN_DATASET=dt, VAL_DATASET=d_val, OUTPUT_PATH=".")
    trainer = Trainer(setting, color_map=map)
    trainer.train()
