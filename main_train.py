# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/jiecaoyu/XNOR-Net-PyTorch
And from https://gist.github.com/daquexian/7db1e7f1e0a92ab13ac1ad028233a9eb
"""
import time
import logging
import os
import sys

# define logger
logging.basicConfig(\
  level=logging.INFO,\
  format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
  handlers=[
  logging.FileHandler("train.log"),
  logging.StreamHandler(sys.stderr)
])

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
from utils.configuration_utils import read_from_config
from dataset.image_folder_lmdb import ImageFolderLMDB

if __name__ == '__main__':
  ## Define some parameters

  if torch.cuda.is_available():
    logging.info('Using CUDA for training')
    
  # define device
  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')

  # define training args and best_acc global variable
  args = read_from_config(os.path.join('config', 'train.json'))
  optimizer_args = args['optimizers'][args['chosen_optimizer']]

  if args['debug']:
    logging.info('Running in debug mode')
    args['train_path'] = args['val_path'] # make smaller
  best_acc = 0

  # define seed
  torch.manual_seed(1)
  torch.cuda.manual_seed(1)

  ## Initialize the datasets and dataloaders for training and testing"
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

  trainset = ImageFolderLMDB(args['train_path'], transform=transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.ToTensor(),
      normalize,
    ])
  )
  if args['debug']:
    trainset = torch.utils.data.Subset(
        trainset, range(args['batch_size'] * 50))

  num_workers=max(torch.cuda.device_count() * 2, 2)
  if sys.platform == 'win32':
    num_workers = 0
  trainloader = torch.utils.data.DataLoader(trainset,
    batch_size=args['batch_size'],
    shuffle=True,
    num_workers=num_workers)

  testset = ImageFolderLMDB(args['val_path'], transform=transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]),
  )
  if args['debug']:
    testset = torch.utils.data.Subset(
        testset, range(args['batch_size'] * 50))
  # multi-processing for loading dataset not supported for now
  testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,\
    num_workers=num_workers)

  ## Define and initialize the model
  model = Net().to(device)
  logging.info('Number of GPUs found: {}'.format(torch.cuda.device_count()))
  if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    logging.info('Found multiple GPUs, running training on all in parallel')
    model = torch.nn.DataParallel(model).cuda()

  logging.debug('Check if parameters in model are in cuda')
  if torch.cuda.is_available():
    for name, module in model.named_modules():
      for parameter in module.parameters():
        if not parameter.is_cuda:
          logging.error('{} has parameters not in CUDA'.format(name))
        break

  # initialize the model weights
  for m in model.modules():
    if isinstance(m, BinaryConv2dKernel):
      m.weight.data.normal_(0, 0.05)
      if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()

  ## Define solver and criterion
  optimizer = None
  if args['chosen_optimizer'] == 'sgd':
    optimizer = optim.SGD(\
        model.parameters(),\
        lr=optimizer_args['base_lr'],\
        momentum=optimizer_args['momentum'],\
        weight_decay=optimizer_args['weight_decay'])
  elif args['chosen_optimizer'] == 'adam':
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args['base_lr'], 
        weight_decay=0.00001)
  criterion = nn.CrossEntropyLoss()

  ## Define the binarization operator
  bin_op = BinOp(model)

  ## Start training
  best_acc = 0
  for epoch in range(1, args['max_epoch'] + 1):
    adjust_learning_rate(optimizer, epoch, args['scheduler_update_list'])
    start_time = time.time()
    train(model, bin_op, trainloader, optimizer,
          criterion, epoch, args['num_classes'])
    logging.info('Time elapsed for epoch: {} min'.format(round((time.time() - start_time) / 60, 2)))
    current_acc = test(model, bin_op, testloader, criterion, args['num_classes'])
    if current_acc > best_acc:
      best_acc = current_acc
      torch.save(model, 'checkpoints/best_model_{}.pth'.format(round(best_acc, 1)))

  ## Stop training
  logging.info('Training Done. Evaluating final model.')
  save_for_evaluation(model, bin_op)
  test(model, bin_op, testloader, criterion, args['num_classes'])

  logging.info('Now exporting binarized model to ONNX.')
  export_model_to_onnx(model, bin_op, 'birealnet18-custom.onnx')