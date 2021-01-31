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
  train_info_file_handler = logging.FileHandler("trainObjectDetection-{}.log".format(script_start_time))
  train_info_file_handler.setLevel(logging.INFO)

  train_error_file_handler = logging.FileHandler("trainObjectDetection_errors-{}.log".format(script_start_time))
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
from torchvision.datasets import CocoDetection
import lightnet as ln

from model import NetObjectDetection
from binary_classes import BinaryConv2dKernel, BinOp
from utils.train_functions import adjust_learning_rate, train_object_detection, save_for_evaluation, save_checkpoint
from utils.test_functions import test_object_detection
from utils.export_function import export_model_to_onnx
from utils.configuration_utils import read_from_config
from dataset.image_folder_lmdb import ImageFolderLMDB
import utils.coco_utils as coco_utils

# coco_datum_transformer = coco_utils.transform_coco_datum_factory((416, 416), coco_utils.coco_labels_dict, 'cuda')

if __name__ == '__main__':
  ## Define some parameters

  if torch.cuda.is_available():
    logging.info('Using CUDA for training')

  # define device
  device = torch.device('cpu')
  if torch.cuda.is_available():
    logging.info('Using CUDA for training')
    device = torch.device('cuda')

  # define seed
  torch.manual_seed(1)
  torch.cuda.manual_seed(1)
  
  # define training args and best_acc global variable
  args = read_from_config(os.path.join('config', 'train_yolo.json'))
  optimizer_args = args['optimizers'][args['chosen_optimizer']]
  
  # Global metrics
  best_test_loss = 0
  epoch_start = 1

  ## Initialize the datasets and dataloaders for training and testing"
  trainset = CocoDetection(os.path.join(args['coco_root_dir'], 'images', 'train2017'),\
    os.path.join(args['coco_root_dir'], 'annotations', 'instances_train2017.json'), transforms = coco_utils.transform_coco_datum)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
  trainloader.collate_fn = coco_utils.collate_fn

  testset = CocoDetection(os.path.join(args['coco_root_dir'], 'images', 'val2017'),\
      os.path.join(args['coco_root_dir'], 'annotations', 'instances_val2017.json'),\
      transforms = coco_utils.transform_coco_datum)
  
  if args['debug']:
    logging.info('Running in debug mode')
    
    trainset = CocoDetection(os.path.join(args['coco_root_dir'], 'images', 'val2017'),\
      os.path.join(args['coco_root_dir'], 'annotations', 'instances_val2017.json'), transforms = coco_utils.transform_coco_datum)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
    trainloader.collate_fn = coco_utils.collate_fn
    
    trainset = torch.utils.data.Subset(
        trainset, range(args['batch_size'] * 5))
    testset = trainset

  # multi-processing for loading dataset not supported for now
  testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])
  testloader.collate_fn = coco_utils.collate_fn

  ## Define and initialize the model
  model = NetObjectDetection().to(device)
  logging.info('Number of GPUs found: {}'.format(torch.cuda.device_count()))
  if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    logging.info('Found multiple GPUs, running training on all in parallel')
    model = torch.nn.DataParallel(model).cuda()

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
  criterion = ln.network.loss.RegionLoss(
    num_classes=coco_utils.coco_num_classes,
    anchors=coco_utils.coco_anchors_simple
  )

  ## Define the binarization operator
  bin_op = BinOp(model)

  ## initialize weights for binary layers if not resuming from somewhere
  if args.get('resume'):
    args['resume'] = os.path.join('checkpoints', str(args['resume']) )
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
    best_test_loss = checkpoint['best_test_loss']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info("=> loaded checkpoint '{}' (epoch {})"
          .format(args['resume'], checkpoint['epoch']))
    del checkpoint

  ## Start training
  for epoch in range(epoch_start, args['max_epoch'] + 1):
    adjust_learning_rate(optimizer, epoch)
    train_object_detection(model, bin_op, trainloader, optimizer, criterion, epoch, args['use_binary'])
    current_test_loss = test_object_detection(model, bin_op, testloader, criterion, args['use_binary'])
    is_best = current_test_loss > best_test_loss
    best_test_loss = max(best_test_loss, current_test_loss)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
    }, is_best, script_start_time, filename='checkpoint-object-detection.pth.tar')

  ## Stop training
  logging.info('Training done, now exporting binarized model to ONNX.')
  export_model_to_onnx(model, bin_op, 'bi-real-yolo.onnx', sample_input_size=(1, 3, 416, 416))

  ## Define post-processing pipeline
  post = ln.data.transform.Compose([
      ln.data.transform.GetBoundingBoxes(coco_utils.coco_num_classes, coco_utils.coco_anchors_simple, 0.4),
      ln.data.transform.NonMaxSuppression(0.3),
      ln.data.transform.TensorToBrambox((416, 416)),
  ])

  i = 0
  for samples in testloader:
    image, scaled_bboxes = samples['images'], samples['scaled_bboxes']
  # for sample in trainloader.dataset:
    output_tensors = model(image).cpu()
    current_targets = scaled_bboxes.cpu()
    loss_value = criterion(output_tensors, current_targets).detach().item()
    logging.info('see loss_value for sample image {}: {}'.format(i, loss_value))

    image = image.squeeze(0)
    current_bramboxes = post(output_tensors)

    bboxes_to_bramboxes = ln.data.transform.TensorToBrambox((416, 416))
    gt_bramboxes = bboxes_to_bramboxes(coco_utils.convert_gt_bboxes_to_prediction_bboxes(scaled_bboxes[0]))
    coco_utils.show_image_predictions(image, current_bramboxes, gt_bramboxes, 'sample-{}.png'.format(i))

    i += 1
    if i == 10:
      break

  # logging.info('Now exporting binarized model to ONNX.')
  # export_model_to_onnx(model, bin_op, 'bi-real-yolo.onnx')