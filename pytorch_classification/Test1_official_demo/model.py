'''
计算公式：经过卷积后的矩阵尺寸大小计算公式
N=(W-F+2P)/S+1，其中w是图片的高度（宽度），f是卷积核的大小，
p是边缘补充的宽度（默认为0），s是步长。所以下面# input(3, 32, 32)  output(16, 28, 28)
中是28.至于16是由于输出维度就等于卷积核个数， self.conv1 = nn.Conv2d(3, 16, 5)
中定义了卷积核个数为16.
'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # input(3, 32, 32)  output(16, 28, 28)
        x = self.pool1(x)           # output(16, 14, 14)
        x = F.relu(self.conv2(x))   # output(32, 10, 10)
        x = self.pool2(x)           # output(32, 5, 5)
        x = x.view(-1, 32*5*5)      # output(32*5*5)
        x = F.relu(self.fc1(x))     # output(120)
        x = F.relu(self.fc2(x))     # output(84)
        x = self.fc3(x)             # output(10)
        return x