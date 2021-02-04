import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryConv2dKernel(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2dKernel, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, bias)

    def forward(self, input):
        # 1. The input of binary convolution shoule be only +1 or -1,
        #    so instead of padding 0 automatically, we need pad -1 by ourselves
        # 2. `padding` of nn.Conv2d is always a tuple of (padH, padW),
        #    while the parameter of F.pad should be (padLeft, padRight, padTop, padBottom)
        # input = F.pad(input, (self.padding[0], self.padding[0],
        #                       self.padding[1], self.padding[1]), mode='constant', value=-1)
        # out = F.conv2d(input, self.weight, self.bias, self.stride,
        #                 0, self.dilation, self.groups, )

        out = super().forward(input)
        return out


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    @staticmethod
    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinActiveBiReal(nn.Module):
    '''
    Binarize the input activations for ***** BiReal  *****.
    '''
    def __init__(self):
        super(BinActiveBiReal, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class BinOp():
    def __init__(self, model):
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m, BinaryConv2dKernel):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)
        assert(len(self.saved_params) == len(self.target_modules))
        self.num_of_params = len(self.saved_params)

    def binarization(self):
        self._meancenterConvParams()
        self._clampConvParams()
        self._save_params()
        self._binarizeConvParams()

    def _meancenterConvParams(self):
        for index in range(self.num_of_params):
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(
                negMean)

    def _clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                self.target_modules[index].data.clamp(-1.0, 1.0)

    def _save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def _binarizeConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = self.target_modules[index].data.sign(
            )

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(
                m_add).mul(1.0-1.0/s[1]).mul(n)
