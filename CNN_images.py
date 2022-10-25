from CNN_image_dataset import CNNImageDataset
from transfer_CNN import TransferCNN
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

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop((112,112)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

num_epochs=20
batch_size=64
lr = 0.001
momentum=0.9
model = TransferCNN()

dataset = CNNImageDataset('data/cl_img_pickle.pkl', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def check_accuracy(loader, model):
    if loader == loader:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on validation data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            scores = model(x)
            predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores])
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    return f"{float(num_correct)/float(num_samples)*100:.2f}"


def train():
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        loop = tqdm(loader, total=len(loader), leave=True)

        if epoch != 0 and epoch % 2 == 0:
            loop.set_postfix(val_acc = check_accuracy(loader, model))

        for features, labels in loop:
            optimiser.zero_grad()

            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimiser.step()

            # running_loss += loss.item() * features.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            print("Loss: ", loss.item())
            writer.add_scalar('loss', loss.item(), batch_idx)

            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())
            batch_idx += 1


if __name__ == '__main__':
    train()
    

