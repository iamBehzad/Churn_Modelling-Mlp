import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpNet1(nn.Module):
    def __init__(self, NumOfFeature, NumOfHiddenLayerNodes, NumOfLabel):
        super(MlpNet1, self).__init__()
        self.fc1 = nn.Linear(NumOfFeature, NumOfHiddenLayerNodes)
        self.fc2 = nn.Linear(NumOfHiddenLayerNodes, NumOfLabel)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MlpNet2(nn.Module):
    def __init__(self, NumOfFeature, NumOfHiddenLayer1Nodes, NumOfHiddenLayer2Nodes, NumOfLabel):
        super(MlpNet2, self).__init__()
        self.fc1 = nn.Linear(NumOfFeature, NumOfHiddenLayer1Nodes)
        self.fc2 = nn.Linear(NumOfHiddenLayer1Nodes, NumOfHiddenLayer2Nodes)
        self.fc3 = nn.Linear(NumOfHiddenLayer2Nodes, NumOfLabel)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MlpNet3(nn.Module):
    def __init__(self, NumOfFeature, NumOfHiddenLayer1Nodes, NumOfHiddenLayer2Nodes, NumOfHiddenLayer3Nodes, NumOfLabel):
        super(MlpNet3, self).__init__()
        self.fc1 = nn.Linear(NumOfFeature, NumOfHiddenLayer1Nodes)
        self.fc2 = nn.Linear(NumOfHiddenLayer1Nodes, NumOfHiddenLayer2Nodes)
        self.fc3 = nn.Linear(NumOfHiddenLayer2Nodes, NumOfHiddenLayer3Nodes)
        self.fc4 = nn.Linear(NumOfHiddenLayer3Nodes, NumOfLabel)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
