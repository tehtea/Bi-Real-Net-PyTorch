# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/jiecaoyu/XNOR-Net-PyTorch
And from https://gist.github.com/daquexian/7db1e7f1e0a92ab13ac1ad028233a9eb
"""
import logging
import argparse

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from torchvision import transforms

from model import Net
from binary_classes import BinaryConv2dKernel, BinOp
from utils.train_functions import adjust_learning_rate, train, save_for_evaluation
from utils.test_functions import test
from utils.export_function import export_model_to_onnx
from dataset.image_folder_lmdb import ImageFolderLMDB


## Define some parameters

# define training args and best_acc global variable
parser = argparse.ArgumentParser(description='Bi-Real-Net 18 Training')
parser.add_argument('--train_path', required=True, help='Path to the train set .lmdb file')
parser.add_argument('--val_path', required=True, help='Path to the validation set .lmdb file')
parser.add_argument('--base_lr', default=0.01, help='base learning rate')
parser.add_argument('--max_epoch', default=50, help='max number of epochs for training')
parser.add_argument('--batch_size', default=64, help='batch size used for training')
parser.add_argument('--num_classes', default=1000, help='Number of classes in dataset')
parser.add_argument('--train_log_file', default='train.log', help='Path to log file for training')
parser.add_argument('--debug', default=True, help='Debug mode to minimally ensure program does not crash')

args = parser.parse_args()
if args.debug:
  args.max_epoch = 1
  args.train_path = args.val_path # make smaller
best_acc = 0

# define logger
logging.basicConfig(filename=args.train_log_file, filemode='w', level=logging.INFO)

# define device
device = torch.device('cpu')
if torch.cuda.is_available():
  logging.info('Using CUDA for training')
  device = torch.device('cuda')

# define seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

## Initialize the datasets and dataloaders for training and testing"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trainset = ImageFolderLMDB(args.train_path, transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]))
if args.debug:
  trainset = torch.utils.data.Subset(
      trainset, range(args.batch_size * 5))
# multi-processing for loading dataset not supported for now
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

testset = ImageFolderLMDB(args.val_path, transform=transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))
if args.debug:
  testset = torch.utils.data.Subset(
      testset, range(args.batch_size * 5))
# multi-processing for loading dataset not supported for now
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

## Define and initialize the model
model = Net().to(device)

# initialize the model weights
for m in model.modules():
  if isinstance(m, BinaryConv2dKernel):
    m.weight.data.normal_(0, 0.05)
    if hasattr(m, 'bias') and m.bias is not None:
      m.bias.data.zero_()

## Define solver and criterion
optimizer = optim.Adam(
    model.parameters(), lr=args.base_lr, weight_decay=0.00001)
criterion = nn.CrossEntropyLoss()

## Define the binarization operator
bin_op = BinOp(model)

## Start training
best_acc = 0
for epoch in range(1, args.max_epoch + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, bin_op, trainloader, optimizer,
          criterion, epoch, args.num_classes)
    current_acc = test(model, bin_op, testloader, criterion, args.num_classes)
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model, 'checkpoints/best_model_{}.pth'.format(round(best_acc, 1)))

## Stop training
logging.info('Training Done. Evaluating final model.')
save_for_evaluation(model, bin_op)
test(model, bin_op, testloader, criterion, args.num_classes)

logging.info('Now exporting binarized model to ONNX.')
export_model_to_onnx(model, bin_op, 'birealnet18-custom.onnx')