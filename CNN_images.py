from turtle import forward
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

class CNNImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_pickle('data/cl_img_pickle.pkl')
        

    def __getitem__(self, index):
        example = self.data.iloc[index]
        feature = example['img_arr']
        label = example['category']
        return (feature, label)

    def __len__(self):
        return len(self.data) 


dataset = CNNImageDataset()
print(dataset[5])
print(len(dataset))

loader = DataLoader(dataset, batch_size=16, shuffle=True)

def train(model, epochs=10):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in loader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels)
            loss.backward()
            print(loss.item)
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1

class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            ## add layers here.
        )
        #define layers

    def forward(self, X):
        return self.layers(X)
        #return prediction

print(next(iter(loader)))