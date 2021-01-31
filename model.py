import torch
import torch.nn as nn

from layers import Conv2dBatch, BinConv2d

"""### Define the actual model"""
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      self.conv1 = Conv2dBatch(3, 64, kernel_size=7, stride=2, padding=3, momentum=1, eps=1e-5)
      self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      self.conv2 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv3 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv4 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv5 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      self.avgPool6a = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv6a = Conv2dBatch(64, 128, kernel_size=1, stride=1, padding=0, momentum=1, eps=1e-5)
      self.conv6b = BinConv2d(64, 128, kernel_size=3, stride=2, padding=1, momentum=1, eps=1e-5)

      self.conv7 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv8 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv9 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      self.avgPool10a = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv10a = Conv2dBatch(128, 256, kernel_size=1, stride=1, padding=0, momentum=1, eps=1e-5)
      self.conv10b = BinConv2d(128, 256, kernel_size=3, stride=2, padding=1, momentum=1, eps=1e-5)

      self.conv11 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv12 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv13 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      self.avgPool14a = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv14a = Conv2dBatch(256, 512, kernel_size=1, stride=1, padding=0, momentum=1, eps=1e-5)
      self.conv14b = BinConv2d(256, 512, kernel_size=3, stride=2, padding=1, momentum=1, eps=1e-5)

      self.conv15 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv16 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv17 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      self.avgPool18 = nn.AvgPool2d(kernel_size=7, stride=1)
      self.conv18 = nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
      if self.train:
        for m in self.modules():
          if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and hasattr(m.weight, 'data'):
            m.weight.data.clamp_(min=0.01)

      x = self.conv1(x)
      x = self.maxPool2(x)
      x = x + self.conv2(x)
      x = x + self.conv3(x)
      x = x + self.conv4(x)
      x = x + self.conv5(x)

      x = (self.conv6a(self.avgPool6a(x)) + self.conv6b(x))
      x = x + self.conv7(x)
      x = x + self.conv8(x)
      x = x + self.conv9(x)

      x = self.conv10a(self.avgPool10a(x)) + self.conv10b(x)
      x = x + self.conv11(x)
      x = x + self.conv12(x)
      x = x + self.conv13(x)

      x = self.conv14a(self.avgPool14a(x)) + self.conv14b(x)
      x = x + self.conv15(x)
      x = x + self.conv16(x)
      x = x + self.conv17(x)

      x = self.avgPool18(x)
      x = self.conv18(x)
      return x

class NetObjectDetection(nn.Module):
    def __init__(self):
      super(NetObjectDetection, self).__init__()

      self.conv1 = Conv2dBatch(3, 64, kernel_size=7, stride=2, padding=3, momentum=1, eps=1e-5)
      self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      self.conv2 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv3 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv4 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv5 = BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      self.avgPool6a = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv6a = Conv2dBatch(64, 128, kernel_size=1, stride=1, padding=0, momentum=1, eps=1e-5)
      self.conv6b = BinConv2d(64, 128, kernel_size=3, stride=2, padding=1, momentum=1, eps=1e-5)

      self.conv7 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv8 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv9 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      self.avgPool10a = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv10a = Conv2dBatch(128, 256, kernel_size=1, stride=1, padding=0, momentum=1, eps=1e-5)
      self.conv10b = BinConv2d(128, 256, kernel_size=3, stride=2, padding=1, momentum=1, eps=1e-5)

      self.conv11 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv12 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv13 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      self.avgPool14a = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv14a = Conv2dBatch(256, 512, kernel_size=1, stride=1, padding=0, momentum=1, eps=1e-5)
      self.conv14b = BinConv2d(256, 512, kernel_size=3, stride=2, padding=1, momentum=1, eps=1e-5)

      self.conv15 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv16 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv17 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)

      # remove these 2 layers from classifier
      # self.avgPool18 = nn.AvgPool2d(kernel_size=7, stride=1)
      # self.conv18 = nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=1, stride=1, padding=0)
      self.conv18 = BinConv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv19 = BinConv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      self.conv20 = BinConv2d(input_channels=512, output_channels=512, kernel_size=3, stride=1, padding=1, momentum=1, eps=1e-5)
      
      self.conv21 = nn.Conv2d(in_channels=512, out_channels=85*5, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
      if self.train:
        for m in self.modules():
          if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and hasattr(m.weight, 'data'):
            m.weight.data.clamp_(min=0.01)

      x = self.conv1(x)
      x = self.maxPool2(x)
      x = x + self.conv2(x)
      x = x + self.conv3(x)
      x = x + self.conv4(x)
      x = x + self.conv5(x)

      x = (self.conv6a(self.avgPool6a(x)) + self.conv6b(x))
      x = x + self.conv7(x)
      x = x + self.conv8(x)
      x = x + self.conv9(x)

      x = self.conv10a(self.avgPool10a(x)) + self.conv10b(x)
      x = x + self.conv11(x)
      x = x + self.conv12(x)
      x = x + self.conv13(x)

      x = self.conv14a(self.avgPool14a(x)) + self.conv14b(x)
      x = x + self.conv15(x)
      x = x + self.conv16(x)
      x = x + self.conv17(x)

      x = x + self.conv18(x)
      x = x + self.conv19(x)
      x = x + self.conv20(x)

      x = self.conv21(x)
      return x
      
if __name__ == "__main__":
  test = Net()
  sample_input = torch.rand((1, 3, 224, 224))
  out = test.forward(sample_input)
  print('see out size: ', out.size())