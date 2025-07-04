import torch


class DataSet(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data     = data
        self.features = data.filter(like="x_").values
        self.labels   = data.filter(like="y_").values


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label   = torch.tensor(self.labels[  idx], dtype=torch.float32)
        return feature, label
