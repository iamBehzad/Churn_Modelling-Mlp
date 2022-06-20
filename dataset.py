import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.nn.functional import normalize

class Churn_ModellingDataset(Dataset):

    def __init__(self, Dataset_Path):

        self.data = pd.read_excel(Dataset_Path)

        self.data = self.data.iloc[:, 3:14]

        self.data.loc[self.data['Geography'] == 'France', 'Geography'] = 0
        self.data.loc[self.data['Geography'] == 'Spain', 'Geography'] = 1
        self.data.loc[self.data['Geography'] == 'Germany', 'Geography'] = 2

        self.data.loc[self.data['Gender'] == 'Male', 'Gender'] = 0
        self.data.loc[self.data['Gender'] == 'Female', 'Gender'] = 1

        # change string value to numeric
        self.data = self.data.apply(pd.to_numeric)

        # change dataframe to array
        self.data = self.data.values

        self.x = torch.Tensor(self.data[:, :10]).float()
        self.y = torch.Tensor(self.data[:, 10]).long()

        # Normalize
        self.x = normalize(self.x, p=2.0, dim=0)

    def __getitem__(self, item):
        sample = (self.x[item, :], self.y[item])
        return sample

    def __len__(self):
        return len(self.data)
