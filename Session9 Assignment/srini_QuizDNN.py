<<<<<<< HEAD
from datetime import datetime
print("Current Date/Time: ", datetime.now())

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, 1, padding=1)    

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)     
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)                

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm5 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(2, 2)                

        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm7 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm8 = nn.BatchNorm2d(64)

        self.pool3 = nn.AvgPool2d(8, 8)                

        self.fc = nn.Linear(64, 10)


    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.batchnorm1(F.relu(self.conv1(x1))) 
        x3 = self.batchnorm2(F.relu(self.conv2(x1 + x2)))
        x4 = self.pool1(x1+x2+x3)
        
        x5 = self.batchnorm3(F.relu(self.conv3(x4)))
        x6 = self.batchnorm4(F.relu(self.conv4(x4+x5)))
        x7 = self.batchnorm5(F.relu(self.conv5(x4+x5+x6)))
        x8 = self.pool2(x5+x6+x7)
        
        x9 = self.batchnorm6(F.relu(self.conv6(x8)))
        x10 = self.batchnorm7(F.relu(self.conv7(x8+x9)))
        x11 = self.batchnorm8(F.relu(self.conv8(x8+x9+x10)))
        x12 = self.pool3(x11)

        x12 = x12.view(-1,64)
        x = self.fc(x12)

        return F.log_softmax(x)

=======
from datetime import datetime
print("Current Date/Time: ", datetime.now())

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, 1, padding=1)    

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)     
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)                

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm5 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(2, 2)                

        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm7 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)    
        self.batchnorm8 = nn.BatchNorm2d(64)

        self.pool3 = nn.AvgPool2d(8, 8)                

        self.fc = nn.Linear(64, 10)


    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.batchnorm1(F.relu(self.conv1(x1))) 
        x3 = self.batchnorm2(F.relu(self.conv2(x1 + x2)))
        x4 = self.pool1(x1+x2+x3)
        
        x5 = self.batchnorm3(F.relu(self.conv3(x4)))
        x6 = self.batchnorm4(F.relu(self.conv4(x4+x5)))
        x7 = self.batchnorm5(F.relu(self.conv5(x4+x5+x6)))
        x8 = self.pool2(x5+x6+x7)
        
        x9 = self.batchnorm6(F.relu(self.conv6(x8)))
        x10 = self.batchnorm7(F.relu(self.conv7(x8+x9)))
        x11 = self.batchnorm8(F.relu(self.conv8(x8+x9+x10)))
        x12 = self.pool3(x11)

        x12 = x12.view(-1,64)
        x = self.fc(x12)

        return F.log_softmax(x)

>>>>>>> 5d248e4e6ce69c748e354d2362a1cfeabcc61bfb
