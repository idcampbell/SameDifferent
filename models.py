import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(32*4*4+2, 16)
        self.fc2 = nn.Linear(16, 4)
        
        #self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.pool = nn.MaxPool2d(2)
        #self.dropout = nn.Dropout()
        #self.fc1 = nn.Linear(500, 32)
        #self.fc2 = nn.Linear(34, 16)
        #self.fc3 = nn.Linear(16, 4)

    def forward(self, x, task):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        z = F.relu(self.fc1(torch.hstack([x, task])))
        x = self.fc2(z)
        
        #x = F.relu(self.pool(self.conv1(x)))
        #x = F.relu(self.pool(self.conv2(x)))
        #x = torch.flatten(x, 1) #self.dropout(torch.flatten(x, 1))
        #x = F.relu(self.fc1(x))
        #z = self.fc2(torch.hstack([x, task]))
        #x = self.fc3(z)
        return x, z
    
class LinearEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32+2, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x, task):
        x = F.relu(self.fc1(torch.hstack([x, task])))
        return self.fc2(x), x
    
class CoRelNet(nn.Module):
    def __init__(self, encoder='linear'):
        super().__init__()
        if encoder=='linear':
            self.enc = LinearEncoder()
        elif encoder=='conv':
            self.enc = ConvEncoder()
        else:
            raise Exception('Invalid encoder type (linear or conv)')
        self.out = nn.Linear(1, 1)
    
    def forward(self, x1, x2, task):
        x1, z1 = self.enc(x1, task) #[:, :2])
        x2, z2 = self.enc(x2, task) #[:, 2:])
        
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
 
        x1_norm = torch.norm(x1, dim=1)
        x2_norm = torch.norm(x2, dim=1)
        x = torch.einsum('bn,bn->b', x1, x2) / (x1_norm * x2_norm)
        x = torch.sigmoid(self.out(x.unsqueeze(1)))
        return x, x1, x2, z1, z2
    
    def loss(self, out, ys):
        #out = torch.clamp(out, min=0, max=1)
        #loss = torch.nn.MSELoss()
        return F.binary_cross_entropy(out, ys.unsqueeze(1)) #loss(out, ys.unsqueeze(1)) #