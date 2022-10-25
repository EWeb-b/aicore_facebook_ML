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


class TransferCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Linear(num_ftrs, 13)

    def forward(self, X):
        return self.resnet50(X)