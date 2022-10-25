from CNN_image_dataset import CNNImageDataset
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 9, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1, 1),
            torch.nn.Dropout2d(0.25),

            torch.nn.Conv2d(9, 27, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1, 1),
            torch.nn.Dropout2d(0.25),

            torch.nn.Flatten(),
            torch.nn.Linear(428652, 13),
            torch.nn.Softmax()
        )

    def forward(self, features):
        return self.layers(features)