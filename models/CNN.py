import torch
from torch import nn
from torch._C import device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, actionSpaceSize, obsSpaceSize):
        super(NeuralNetwork, self).__init__()
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        

        self.fc1 = nn.Linear(3136, 512)
        torch.nn.init.kaiming_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(512, self.actionSpaceSize)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

        self.relu = nn.ReLU(inplace=True)
        
    

    def forward(self, x):
        x = x.to(device)/255
        

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten output
        x = x.view(x.size()[0], -1)
        

        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        
        return x