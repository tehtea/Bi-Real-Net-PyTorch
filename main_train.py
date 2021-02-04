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

script_start_time = int(time.time())

if __name__ == '__main__':
  train_info_file_handler = logging.FileHandler("train-{}.log".format(script_start_time))
  train_info_file_handler.setLevel(logging.INFO)

  train_error_file_handler = logging.FileHandler("train_errors-{}.log".format(script_start_time))
  train_error_file_handler.setLevel(logging.ERROR)

  logging.basicConfig(\
    level=logging.INFO,\
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    handlers=[train_info_file_handler, train_error_file_handler]\
  )

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from torchvision import transforms

from model import Net
from binary_classes import BinaryConv2dKernel, BinOp
from utils.train_functions import adjust_learning_rate, train, save_for_evaluation, save_checkpoint
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

  # Global metrics
  best_top_1_acc = 0
  epoch_start = 1

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

  num_workers=args['num_workers']
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
  testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False,\
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
        lr=optimizer_args['base_lr'], 
        weight_decay=optimizer_args['weight_decay'])
  criterion = nn.CrossEntropyLoss()

  ## Define the binarization operator
  bin_op = BinOp(model)

  if args.get('resume'):
    args['resume'] = os.path.join('checkpoints', str(args['resume']) )

  # initialize weights for binary layers if not resuming from somewhere
  if not args.get('resume') or not os.path.isfile(args['resume']):
    logging.info('Initializing weights from scratch')
    for name, module in model.named_modules():
      if isinstance(module, BinaryConv2dKernel):
        module.weight.data.uniform_(-0.0005, 0.0005)
        if hasattr(module, 'bias') and module.bias is not None:
          module.bias.data.zero_()
  else:
    logging.info("=> loading checkpoint '{}'".format(args['resume']))
    checkpoint = torch.load(args['resume'])
    epoch_start = checkpoint['epoch']
    best_top_1_acc = checkpoint['best_top_1_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info("=> loaded checkpoint '{}' (epoch {})"
          .format(args['resume'], checkpoint['epoch']))
    del checkpoint

  ## Start training
  best_top_1_acc = 0
  current_epoch = epoch_start
  while current_epoch < args['max_epoch'] + 1:
    adjust_learning_rate(optimizer, current_epoch, args['scheduler_update_list'])

    start_time = time.time()
    progress_one_epoch = train(model, bin_op, trainloader, optimizer,
          criterion, current_epoch, args['num_classes'], args['use_binary'])
    if (not progress_one_epoch):
      continue
    logging.info('Time elapsed for epoch: {} min'.format(round((time.time() - start_time) / 60, 2)))
    
    current_top_1_acc = test(model, bin_op, testloader, criterion, args['num_classes'], args['use_binary'])
    is_best = current_top_1_acc > best_top_1_acc
    best_top_1_acc = max(best_top_1_acc, current_top_1_acc)
    save_checkpoint({
        'epoch': current_epoch,
        'state_dict': model.state_dict(),
        'best_top_1_acc': best_top_1_acc,
        'optimizer' : optimizer.state_dict(),
    }, is_best, script_start_time)

    current_epoch += 1

  ## Stop training
  logging.info('Training done, now exporting binarized model to ONNX.')
  export_model_to_onnx(model, bin_op, 'birealnet18-custom.onnx')