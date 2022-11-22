import copy
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from text_classifier_model import TextClassifier
from text_dataset import TextDataset

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 25        
batch_size=64
lr = 0.0001
max_length = 200
momentum=0.9
shuffle = True
num_workers=1
pin_memory=True
model = TextClassifier(num_classes = 13)
model = model.to(device)
# optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimiser = torch.optim.Adam(model.main.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.9)

dataset = TextDataset(labels_level=0, max_length=max_length)
num_train_data = len(dataset) - round((len(dataset) * 0.25))
num_val_data = len(dataset) - num_train_data
train_set, validation_set = random_split(dataset,[num_train_data, num_val_data])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
val_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)


def train():
    best_model_weights = copy.deepcopy(model.main.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("-~-" * 30)  
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print("-~-" * 30)        

        trn_run_loss2 = 0.0
        trn_run_loss = 0.0
        trn_run_corr = 0
        # Training loop.
        for features, labels in train_loader:
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

            trn_run_loss2 += loss.item()
            trn_run_loss += loss.item() * features.size(0)
            trn_run_corr += torch.sum(preds == labels.data)

        # exp_lr_scheduler.step()
        trn_epoch_loss = trn_run_loss / num_train_data
        trn_epoch_acc = (trn_run_corr.double() / num_train_data) * 100
        print(f' Epoch [{epoch + 1}/{num_epochs}] TRAINING Loss: {trn_epoch_loss:.4f}, Acc: {trn_epoch_acc:.4f}%')

        val_run_loss = 0.0
        val_run_corr = 0
        correct = 0.0
        # Eval loop
        with torch.no_grad():
            for features, labels in val_loader:
                model.eval()
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_run_loss += loss.item() * features.size(0)
                val_run_corr += torch.sum(preds == labels.data)
                correct += (preds == labels).sum().item()
        
        val_epoch_loss = val_run_loss / num_val_data
        val_epoch_acc = (val_run_corr.double() / num_val_data) * 100
        print(f' Epoch [{epoch + 1}/{num_epochs}] VALIDATION Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}%')

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_weights = copy.deepcopy(model.main.state_dict())
            print(f"New best accuracy! {best_acc:.4f}")
        print("\n")

    print(f"Best val accuracy: {best_acc:4f}")
    # We set strict=False because we only want to save the model.main state dict, but load_state_dict expects the model.bert parameters too.
    model.load_state_dict(best_model_weights, strict=False) 
    torch.save(model.main.state_dict(), 'text_model.pt')


if __name__ == '__main__':
    print(device)
    train()

    