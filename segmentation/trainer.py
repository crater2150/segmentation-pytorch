from segmentation.dataset import dirs_to_pandaframe, load_image_map_from_file, MaskDataset, compose, post_transforms
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(model, device, train_loader, optimizer, epoch, accumulation_steps=8):
    model.train()
    total_train = 0
    correct_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss = loss / accumulation_steps
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        total_train += target.nelement()
        correct_train += predicted.eq(target.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                         len(train_loader.dataset),
                                                                                         100. * batch_idx / len(
                                                                                             train_loader), loss.item(),
                                                                                         train_accuracy), end="", flush=True)
        if (batch_idx + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            model.zero_grad()  # Reset gradients tensors


def get_model(classes):
    from segmentation.modules import Architecture
    architecture = Architecture.UNET.get_architecture()('resnet34', classes=classes, activation='logsoftmax')
    return architecture


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
    dt = MaskDataset(a, map, transform=compose([post_transforms()]))
    d_test = MaskDataset(b, map, transform=compose([post_transforms()]))

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from segmentation.model import UNet, AttentionUnet

    model1 = UNet(in_channels=3,
                  out_channels=8,
                  n_class=len(map),
                  kernel_size=3,
                  padding=1,
                  stride=1)

    model2 = AttentionUnet(
        in_channels=3,
        out_channels=8,
        n_class=len(map),
        kernel_size=3,
        padding=1,
        stride=1,
        attention=True)

    model = get_model(len(map))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.NLLLoss()
    model.float()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    from torch.utils import data

    train_loader = data.DataLoader(dataset=dt, batch_size=1, shuffle=True, num_workers=5)
    test_loader = data.DataLoader(dataset=d_test, batch_size=1, shuffle=False)


    for epoch in range(1, 3):
        print('Training started ...')
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
