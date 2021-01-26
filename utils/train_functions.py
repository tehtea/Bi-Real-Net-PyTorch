import logging
import threading, queue

import torch
from torch.autograd import Variable

def adjust_learning_rate(optimizer, epoch, update_list=[120, 200, 240, 280]):
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

# Train model for one epoch
def train(model, bin_op, trainloader, optimizer, criterion, epoch, num_classes):
    model.train()

    # queue for placing pre-processed training samples in background
    _train_samples_queue = queue.Queue(maxsize=max(torch.cuda.device_count() * 4, 4))

    # function to be run by the preprocessing daemon
    def _put_in_queue():
        for batch_idx, (data, target) in enumerate(trainloader):
            _train_samples_queue.put((batch_idx, (data, target)))

    # start the daemon for processing training samples
    producer = threading.Thread(target=_put_in_queue, daemon=True)
    producer.start()

    num_batches = 0
    while num_batches != len(trainloader):
        # Get train samples from queue
        batch_idx, (data, target) = _train_samples_queue.get()

        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        output = output.view(output.size(0), num_classes)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        # Run one step of optimizer
        optimizer.step()

        # log progress every 100 batches
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
        
        # Inform the queue that a sample has been processed
        _train_samples_queue.task_done()

        # [DEBUGGING] add counter for number of batches processed
        num_batches += 1
    logging.info('Number of batches processed in epoch: {}'.format(num_batches))

def save_for_evaluation(model, bin_op):
    model.eval()
    bin_op.binarization()
    torch.save(model, 'checkpoints/binarized_model.pb')
    bin_op.restore()