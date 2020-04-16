# from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

TRAIN_DIR = 'traffic-sign/train'
TEST_DIR = 'traffic-sign/test'
VAL_DIR = 'traffic-sign/val_images'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch 62 Traffic Sign Recognition Train')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor

# 图像增广
from data import data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, \
    data_jitter_contrast, data_rotate, data_hvflip, data_shear, data_translate, data_center, data_hflip, \
    data_vflip


# Apply data transformations on the training images to augment dataset
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_transforms),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_jitter_brightness),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_jitter_hue),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_jitter_contrast),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_jitter_saturation),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_translate),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_rotate),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_hvflip),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_center),
                                    datasets.ImageFolder(TRAIN_DIR,
                                                         transform=data_shear)]), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(VAL_DIR, transform=data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)

# Neural Network and Optimizer
from model import Net

model = Net()

if use_gpu:
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)


def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim=1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item() / (args.batch_size * args.log_interval),
                loss.data.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss, 2))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = 'model_' + str(epoch) + '.pth'
    save_model_name = os.path.join(args.save_directory, model_file)
    torch.save(model.state_dict(), save_model_name)
    print('\nSaved model to ' + model_file)
