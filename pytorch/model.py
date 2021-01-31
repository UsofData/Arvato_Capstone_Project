import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
                
        # defining linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fch = nn.Linear(hidden_dim, hidden_dim)
        self.fch2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.2)
        # sigmoid layer
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        out = F.relu(self.fc1(x)) # activation on hidden layer
        out = self.drop(out)
        out = F.relu(self.fch(out))
        out = self.drop(out)
        out = F.relu(self.fch2(out))
        out = self.drop(out)
        out = self.fc2(out)
        return self.sig(out) # returning class score