from image_classification.CNN_image_dataset import CNNImageDataset
from image_classification.image_transfer_CNN_model import ImageTransferCNN

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split 
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms

transform = transforms.Compose(
        [
            transforms.ToTensor(), # This also changes the pixels to be in range [0, 1] from [0, 255].
            # transforms.RandomResizedCrop((64,64))
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    
device = ("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
num_epochs=20
batch_size=128
lr = 0.001
shuffle = True
num_workers=1
pin_memory=True

model = ImageTransferCNN()
model = model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

dataset = CNNImageDataset('data/cl_img_pickle_final.zip', transform=transform)
num_train_data = len(dataset) - round((len(dataset) * 0.25))
num_val_data = len(dataset) - num_train_data
train_set, validation_set = random_split(dataset,[num_train_data, num_val_data])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
val_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

def train():
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("-~-" * 30)  
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print("-~-" * 30)        

        trn_run_loss2 = 0.0
        trn_run_loss = 0.0
        trn_run_corr = 0
        
        # Training loop.
        for batch_index, (features, labels) in enumerate(train_loader, 0):
            model.train()
            features = features.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            torch.set_grad_enabled(True)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()

            trn_run_loss += loss.item() * features.size(0)
            correct = torch.sum(preds == labels.data)
            trn_run_corr += correct

            writer.add_scalar("Loss (training)", loss.item(), batch_index)
            writer.add_scalar("Accuracy (training)", ((correct / len(preds)) * 100).item(), batch_index)

        trn_epoch_loss = trn_run_loss / num_train_data
        trn_epoch_acc = (trn_run_corr.double() / num_train_data) * 100
        print(f' Epoch [{epoch + 1}/{num_epochs}] TRAINING Loss: {trn_epoch_loss:.4f}, Acc: {trn_epoch_acc:.4f}%')

        writer.add_scalar("Epoch Loss (training)", trn_epoch_loss, epoch)
        writer.add_scalar("Epoch Accuracy (training)", trn_epoch_acc, epoch)

        val_run_loss = 0.0
        val_run_corr = 0

        # Eval loop
        for batch_index, (features, labels) in enumerate(val_loader, 0):
            model.eval()
            features = features.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            torch.set_grad_enabled(False)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_run_loss += loss.item() * features.size(0)
            correct = torch.sum(preds == labels.data)
            val_run_corr += correct

            writer.add_scalar("Loss (validation)", loss.item(), batch_index)
            writer.add_scalar("Accuracy (validation)", ((correct / len(preds)) * 100).item(), batch_index)
        
        val_epoch_loss = val_run_loss / num_val_data
        val_epoch_acc = (val_run_corr.double() / num_val_data) * 100
        print(f' Epoch [{epoch + 1}/{num_epochs}] VALIDATION Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}%')

        writer.add_scalar("Epoch Loss (validation)", val_epoch_loss, epoch)
        writer.add_scalar("Epoch Accuracy (validation)", val_epoch_acc, epoch)

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"New best accuracy! {best_acc:.4f}")
        print("\n")

    print(f"Best val accuracy: {best_acc:4f}")
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict, 'image_model.pt')

if __name__ == '__main__':
    train()
    
    

