import logging

import torch
from torch.autograd import Variable
import queue
import threading

def test(model, bin_op, testloader, criterion, num_classes, use_binary):
    model.eval()
    
    # queue for placing pre-processed testing samples in background
    _test_samples_queue = queue.Queue(maxsize=max(torch.cuda.device_count() * 4, 4))

    # function to be run by the preprocessing daemon
    def _put_in_queue():
        for data, target in testloader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            _test_samples_queue.put((data, target))

    # start the daemon for processing testing samples
    producer = threading.Thread(target=_put_in_queue, daemon=True)
    producer.start()

    num_batches = 0
    test_loss = 0
    top_1_correct = 0
    top_5_correct = 0

    if use_binary:
        bin_op.binarization()
    
    while num_batches != len(testloader):
        # Get test samples from queue
        try:
            data, target = _test_samples_queue.get(timeout=60)
        except queue.Empty:
            logging.error('Did not get new test sample within 60 seconds, skipping current test')
            break

        data, target = Variable(data), Variable(target)
                                    
        output = model(data)
        output = output.view(output.size(0), num_classes)

        test_loss += criterion(output, target).data.item()
        
        _, pred = output.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in (1, 5):
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            if k == 1:
                top_1_correct += correct_k
            elif k == 5:
                top_5_correct += correct_k

        # Inform the queue that a sample has been processed
        _test_samples_queue.task_done()

        # [DEBUGGING] add counter for number of batches processed
        num_batches += 1

    if use_binary:
        bin_op.restore()
    
    top_1_correct = int(top_1_correct)
    top_5_correct = int(top_5_correct)
    top_1_acc = 100. * top_1_correct / len(testloader.dataset)
    top_5_acc = 100. * top_5_correct / len(testloader.dataset)
    
    test_loss /= len(testloader.dataset)
    logging.info('\nTest set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.2f}%),\
        Top-5 Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, 
        top_1_correct, 
        len(testloader.dataset),
        top_1_acc,
        top_5_correct,
        len(testloader.dataset),
        top_5_acc)
    )
    model.train()
    return top_1_acc

def test_object_detection(model, bin_op, testloader, criterion, use_binary):
    # Set model to eval mode
    model.eval()

    if use_binary:
        # Binarize weights in model
        bin_op.binarization()

    # queue for placing pre-processed testing samples in background
    _test_samples_queue = queue.Queue(maxsize=max(torch.cuda.device_count() * 4, 4))

    # function to be run by the preprocessing daemon
    def _put_in_queue():
        for samples in testloader:
            if torch.cuda.is_available():
                samples['images'] = samples['images'].cuda()
                samples['scaled_bboxes'] = samples['scaled_bboxes'].cuda()
            _test_samples_queue.put(samples)

    # start the daemon for processing testing samples
    producer = threading.Thread(target=_put_in_queue, daemon=True)
    producer.start()

    test_loss = 0
    num_batches = 0
    while num_batches != len(testloader):
        # Get test samples from queue
        try:
            samples = _test_samples_queue.get(timeout=60)
            images, scaled_bboxes = samples['images'], samples['scaled_bboxes']
        except queue.Empty:
            logging.error('Did not get new test sample within 60 seconds, skipping current epoch')
            break

        # pass images to model
        output_tensors = model(images)

        # Accumulate the loss
        test_loss += criterion(output_tensors.cpu(), scaled_bboxes.cpu()).item()

        # Inform the queue that a sample has been processed
        _test_samples_queue.task_done()

        # add counter for number of batches processed
        num_batches += 1

    if use_binary:
        # Restore the binary weights
        bin_op.restore()

    # Set model back to training mode
    model.train()

    test_loss /= len(testloader.dataset)
    logging.info('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_loss