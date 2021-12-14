#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier


class NeuralNet(nn.Module):
    def __init__(self, num_features, num_units=10, dropout=0.1):
        super(NeuralNet, self).__init__()
        self.num_units = num_units
        self.num_features = num_features
        self.linear_1 = nn.Linear(num_features, num_units)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_units, num_units)
        self.linear_3 = nn.Linear(num_units, 2)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.linear_3(x)
        x = F.softmax(x, dim=-1)
        return x




if __name__ == "__main__":
    if torch.cuda.is_available():
        print('GPU support enabled')
    else:
        print('WARNING: GPU support not enabled')

    nn = NeuralNetClassifier(NeuralNet, max_epochs=10, lr=0.01, batch_size=12, optimizer=optimRMSprop)
    