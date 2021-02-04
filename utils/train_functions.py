import logging
import threading, queue
import shutil
import os

import torch
from torch.autograd import Variable

def adjust_learning_rate(optimizer, epoch, update_list=[120, 200, 240, 280]):
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

# Train model for one epoch
def train(model, bin_op, trainloader, optimizer, criterion, epoch, num_classes, use_binary):
    model.train()

    progress_one_epoch = True

    # queue for placing pre-processed training samples in background
    _train_samples_queue = queue.Queue(maxsize=max(torch.cuda.device_count() * 4, 4))

    # function to be run by the preprocessing daemon
    def _put_in_queue():
        for batch_idx, (data, target) in enumerate(trainloader):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            _train_samples_queue.put((batch_idx, (data, target)))

    # start the daemon for processing training samples
    producer = threading.Thread(target=_put_in_queue, daemon=True)
    producer.start()

    num_batches = 0
    total_loss = 0
    while num_batches != len(trainloader):
        # Get train samples from queue
        try:
            batch_idx, (data, target) = _train_samples_queue.get(timeout=60)
        except queue.Empty:
            logging.error('Did not get new train sample within 60 seconds, skipping current epoch')
            progress_one_epoch = False
            break

        # process the weights including binarization
        if use_binary:
            bin_op.binarization()
        
        # forwarding

        optimizer.zero_grad()
        output = model(data)
        output = output.view(output.size(0), num_classes)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        if use_binary:
            bin_op.restore()
            bin_op.updateBinaryGradWeight()
        
        # Run one step of optimizer
        optimizer.step()

        # accumulate loss
        total_loss += loss.data.item()

        # log progress every 100 batches
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))

        # source: https://discuss.pytorch.org/t/best-practices-for-maximum-gpu-utilization/13863/5
        # clear loss graph after stepping through optimizer and logging progress 
        del loss

        # Inform the queue that a sample has been processed
        _train_samples_queue.task_done()

        # [DEBUGGING] add counter for number of batches processed
        num_batches += 1
    average_loss = total_loss / len(trainloader.dataset)
    logging.info('Number of batches processed in epoch: {}, average loss: {:.4f}'.format(num_batches, average_loss))

    return progress_one_epoch
    
# Train object detection model for one epoch
def train_object_detection(model, bin_op, trainloader, optimizer, criterion, epoch, use_binary):
    # set model to training mode
    model.train()

    # set the running quantities to zero at the beginning of the epoch
    running_loss=0

    # queue for placing pre-processed training samples in background
    _train_samples_queue = queue.Queue(maxsize=max(torch.cuda.device_count() * 4, 4))

    # function to be run by the preprocessing daemon
    def _put_in_queue():
        for batch_idx, samples in enumerate(trainloader):
            if torch.cuda.is_available():
                samples['images'] = samples['images'].cuda()
                samples['scaled_bboxes'] = samples['scaled_bboxes'].cuda()
            _train_samples_queue.put((batch_idx, samples))

    # start the daemon for processing training samples
    producer = threading.Thread(target=_put_in_queue, daemon=True)
    producer.start()

    num_batches = 0
    while num_batches != len(trainloader):
        # Get train samples from queue
        try:
            batch_idx, samples = _train_samples_queue.get(timeout=60)
        except queue.Empty:
            logging.error('Did not get new train sample within 60 seconds, skipping current epoch')
            break

        # process the weights including binarization
        if use_binary:
            bin_op.binarization()

        # Set the gradients to zeros
        optimizer.zero_grad()

        # get the images and scaled_bboxes from each minibatch
        images, scaled_bboxes = samples['images'], samples['scaled_bboxes']

        # sometimes the minibatch can be truncated. get the actual batch size first
        batch_size = len(images)

        # pass images to model
        output_tensors = model(images)

        # compute loss
        loss_in_batch = criterion(output_tensors.cpu(), scaled_bboxes.cpu())

        # compute gradients
        loss_in_batch.backward()

        # restore weights
        if use_binary:
            bin_op.restore()
            bin_op.updateBinaryGradWeight()
        
        # do one step of the optimizer
        optimizer.step()
        
        # Detach loss value from graph
        current_loss_in_batch = loss_in_batch.detach().item()

        # add the loss of this batch to the running loss
        running_loss += current_loss_in_batch

        # add logging for debug purposes
        if batch_idx % 100 == 0:
            logging.info('at batch {}, see batch size: {}, see loss_in_batch: {}'\
                .format(batch_idx, batch_size, current_loss_in_batch))

        # Inform the queue that a sample has been processed
        _train_samples_queue.task_done()

        # add counter for number of batches processed
        num_batches += 1

    # and compute stats for the full training set
    total_loss = running_loss / len(trainloader.dataset)
    for param_group in optimizer.param_groups:
        logging.info('epoch={}\t lr={}\t loss={}'.format(epoch, param_group['lr'], total_loss))

def save_for_evaluation(model, bin_op):
    model.eval()
    bin_op.binarization()
    torch.save(model, 'checkpoints/binarized_model.pb')
    bin_op.restore()

def save_checkpoint(state, is_best, script_start_time, filename='checkpoint.pth.tar'):
    filename = '{}-{}'.format(script_start_time, filename)
    torch.save(state, os.path.join('checkpoints', filename))
    if is_best:
        shutil.copyfile(os.path.join('checkpoints', filename), os.path.join('checkpoints', '{}-{}'.format(script_start_time, 'model_best.pth.tar')))