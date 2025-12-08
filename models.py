import torch
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.linear1 = nn.Linear(num_features, 100)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(100, 2)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.output(x)
        return x