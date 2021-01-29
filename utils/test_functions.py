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
        
        pred = output.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).cpu().sum()

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
    
    top_1_acc = 100. * float(top_1_correct) / len(testloader.dataset)
    top_5_acc = 100. * float(top_5_correct) / len(testloader.dataset)
    
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