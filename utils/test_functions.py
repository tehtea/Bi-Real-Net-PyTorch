import logging

import torch
from torch.autograd import Variable
import queue
import threading

def test(model, bin_op, testloader, criterion, num_classes):
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
    correct = 0
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
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # Inform the queue that a sample has been processed
        _test_samples_queue.task_done()

        # [DEBUGGING] add counter for number of batches processed
        num_batches += 1

    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)
    
    test_loss /= len(testloader.dataset)
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    model.train()
    return acc