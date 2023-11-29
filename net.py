import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv3(nn.Module):
    def __init__(self, classes):
        super(SimpleConv3, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,2,1) ##3*3卷积，步长为2，padding为1
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc1 = nn.Linear(2304,100)
        self.fc2 = nn.Linear(100, classes)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        # x = x.view(-1,self.num_flat_features(x))
        x = x.view(-1, 2304)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        return x

if __name__ == '__main__':
    input = torch.rand((3,48,48))
    net = SimpleConv3(2)
    output = net(input)
    print(output)