import torch.nn as nn
import torch.nn.functional as F

from binary_classes import BinaryConv2dKernel, BinActive

class Conv2dBatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 momentum=0.1, eps=1e-4, affine=True):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,
                      self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels, momentum=self.momentum,
                           eps=self.eps, affine=self.affine),
        )

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

"""### Define one BinConv2d layer"""
class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, dropout=0, momentum=1, eps=1e-5):
        super(BinConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.conv = BinaryConv2dKernel(input_channels, output_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(
            output_channels, eps=eps, momentum=momentum, affine=True)
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)

    def forward(self, x):
        if self.training:
          x, mean = BinActive.apply(x)
        else:
          # We clone x here because it causes unexpected behaviors
          # to edit the data of `x` tensor.
          # and we cannot do x = x.sign() directly because onnx opset 7 does not have the sign operator
          x = x.clone()
          x.data = x.sign()

        if self.dropout_ratio != 0 and self.training:
          x = F.dropout(x, self.dropout_ratio)
        x = self.conv(x)
        x = self.bn(x)
        return x
