import logging

import torch
from torch.autograd import Variable

# TODO: parameterize the update list
def adjust_learning_rate(optimizer, epoch, update_list=[120, 200, 240, 280]):
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

# Train model for one epoch
def train(model, bin_op, trainloader, optimizer, criterion, epoch, num_classes):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

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
        
        optimizer.step()
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))

def save_for_evaluation(model, bin_op):
    model.eval()
    bin_op.binarization()
    torch.save(model, 'checkpoints/binarized_model.pb')
    bin_op.restore()