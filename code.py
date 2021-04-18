# Machine Learning Assignment 4
# DEVELOPED BY Tomer Himi & CUCUMBER AN OrSN COMPANY.
# UNAUTHORIZED COPY OF THIS WORK IS STRICTLY PROHIBITED.
# DEVELOPED FOR EDUCATIONAL PURPOSES, FOR THE COURSE MACHINE LEARNING 89511.
# BAR ILAN UNIVERSITY, DECEMBER, 2020.
# ALL RIGHTS RESERVED.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets

class AToBNetworks(nn.Module):
    #class for the first two models A and B, using RelU in forward
    def __init__(self, image_size):
        super(AToBNetworks, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)  # Input -> 1st
        self.fc1 = nn.Linear(100, 50)  # 1st -> 2nd
        self.fc2 = nn.Linear(50, 10)  # 2nd -> Output

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim = 1)

class CNetwork(nn.Module):
    #class for the third model using RelU and dropout regularization
    def __init__(self, image_size, fc0_size=100, fc1_size=50, fc2_size=10):
        super(CNetwork, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.do0 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.do0(x)  # Regularization AFTER - do0 stands for Dropout0
        x = F.relu(self.fc1(x))
        x = self.do1(x)  # Regularization AFTER - do1 stands for Dropout1
        return F.log_softmax(self.fc2(x), dim = 1)

class DNetwork(nn.Module):
    #class for the forth model using RelU and batch normalization
    def __init__(self, image_size, fc0_size=100, fc1_size=50, fc2_size=10):
        super(DNetwork, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.bn0 = nn.BatchNorm1d(num_features=fc0_size)  # nn.BatchNorm1d performs Batch normalization.
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.bn1 = nn.BatchNorm1d(num_features=fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.BatchNorm1d(num_features=fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        return F.log_softmax(self.bn2(self.fc2(x)), dim = 1)

class EToFNetworks(nn.Module):
    #class for the final two models E and F using either RelU or Sigmoid and ADAM or SGD
    def __init__(self, image_size, activation):
        super(EToFNetworks, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)  # Input -> 1st
        self.fc1 = nn.Linear(128, 64)  # 1st -> 2nd
        self.fc2 = nn.Linear(64, 10)  # 2nd -> 3rd
        self.fc3 = nn.Linear(10, 10)  # 3rd -> 4th
        self.fc4 = nn.Linear(10, 10)  # 4th -> 5th
        self.fc5 = nn.Linear(10, 10)  # 5th -> Output
        self.activation = activation

    def forward(self, x):
        x = x.view(-1, self.image_size)
        if self.activation == "relu":  # For Fifth model
            x = F.relu(self.fc0(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))

        elif self.activation == "sigmoid":  # RELEVANT TO MODEL 6 ONLY
            x = torch.sigmoid(self.fc0(x))
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            x = torch.sigmoid(self.fc5(x))
        return F.log_softmax(x, dim = 1)

def train(model):
    """Train function, trains our model using the ordinary drill of Forward, Backwards etc. 
    After each epoch of training, validation step has activeted"""
    model.train()  #model is one of the four classes above (AToB, C, D or EToF)
    correct = 0
    train_loss = 0
    val_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):  #train step
        optimizer.zero_grad()
        output = model(data)  #performs Forward apparently
        loss = F.nll_loss(output, labels)
        train_loss += F.nll_loss(output, labels, reduction = 'sum').item()
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim = True)[1]  #get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
        
    for data, labels in test_loader: #val step
        output = model(data)  
        loss = F.nll_loss(output, labels)
        val_loss += F.nll_loss(output, labels, reduction = 'sum').item() 
        
    train_loss /= len(train_loader.dataset)
    val_loss /= len(test_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    val_accuracy = 100. * correct / len(test_loader.dataset)

def test(model, test_data):
    """Test function, helps us predict the label of each example of fashion MNIST (test from PyTorch)"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_data:
            output = model(data)  # Performs Forward apparently
            test_loss += F.nll_loss(output, labels, reduction = 'sum').item()  # sum up batch loss
            pred = output.max(1, keepdim = True)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
        test_loss /= len(test_data.dataset)
        test_accuracy = 100. * correct / len(test_data.dataset)

if __name__ == "__main__":
    transforms = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize((0.5,), (0.5,))])
    fashion = datasets.FashionMNIST("./data", train = True, download = True, transform = transforms)
    train_set, val_set = torch.utils.data.random_split(fashion, [round(len(fashion) * 0.8),
                                                             len(fashion) - round(len(fashion) * 0.8)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size = 64, shuffle = True)
    test_input = torch.utils.data.DataLoader(datasets.FashionMNIST("./data", train = False, transform = transforms),
                                         batch_size = 64, shuffle = False)
    
    #the best model to run
    model = DNetwork(image_size = 28 * 28)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)  #lr stands for Learning Rate. torch.optim injects an OPTIMIZER.
    epoch = 10
        
    #train and validation steps
    for _ in range(epoch):
        train(model)
    
    #test step (test_x file)
    test_xx = torch.FloatTensor(np.loadtxt('test_x') / 255) 
    output = model(test_xx)
    pred = output.max(1, keepdim = True)[1]
    np.savetxt('test_y', pred.numpy(), fmt = '%i')
    
    #test step (PyTorch test_input)
    test(model, test_input)