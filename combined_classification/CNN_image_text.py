import copy
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split 

from combined_classification.image_text_dataset import ImageTextDataset
from text_classification.text_classifier_all import TextClassifier
from image_classification.image_transfer_CNN_model import ImageTransferCNN

class ImageTextClassifier(nn.Module):
    def __init__(self, input_size: int = 768, num_classes:int = 13) -> None:
        super().__init__()
        # self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # num_ftrs = self.resnet50.fc.in_features
        # self.resnet50.fc = torch.nn.Linear(num_ftrs, 13)
        self.image_classifier = ImageTransferCNN()
        self.text_classifier = TextClassifier(input_size=input_size, num_classes=num_classes)
        self.combiner = nn.Linear(26, num_classes)

    def forward(self, image_features, text_features):
        img_result = self.image_classifier(image_features)
        text_result = self.text_classifier(text_features)
        combined = torch.cat((img_result, text_result), 1)
        combined_result = self.combiner(combined)
        return combined_result

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_epochs = 15        
batch_size=64
lr = 0.001
momentum=0.9
shuffle = True
num_workers=1
pin_memory=True
model = ImageTextClassifier(num_classes = 13)
model = model.to(device)
optimiser = torch.optim.Adam([*model.image_classifier.parameters(), *model.text_classifier.main.parameters(), *model.combiner.parameters()], lr=lr)
# optimiser = torch.optim.Adam((model.text_classifier.main.parameters(), model.combiner.parameters()), lr=lr)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.9)

dataset = ImageTextDataset(labels_level=0, max_length=100, read_csv=True)
num_train_data = len(dataset) - round((len(dataset) * 0.25))
num_val_data = len(dataset) - num_train_data
train_set, validation_set = random_split(dataset,[num_train_data, num_val_data])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
val_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

def train():
    best_img_w = copy.deepcopy(model.image_classifier.state_dict())
    best_text_w = copy.deepcopy(model.text_classifier.main.state_dict())
    best_comb_w = copy.deepcopy(model.combiner.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("-~-" * 30)  
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print("-~-" * 30)        

        trn_run_loss2 = 0.0
        trn_run_loss = 0.0
        trn_run_corr = 0
        # Training loop.
        for images, text, labels in train_loader:
            model.train()
            images = images.to(device)
            text = text.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            torch.set_grad_enabled(True)
            outputs = model(images, text)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()

            trn_run_loss2 += loss.item()
            trn_run_loss += loss.item() * images.size(0)
            trn_run_corr += torch.sum(preds == labels.data)

        # exp_lr_scheduler.step()
        trn_epoch_loss = trn_run_loss / num_train_data
        trn_epoch_acc = (trn_run_corr.double() / num_train_data) * 100
        print(f' Epoch [{epoch + 1}/{num_epochs}] TRAINING Loss: {trn_epoch_loss:.4f}, Acc: {trn_epoch_acc:.4f}%')

        val_run_loss = 0.0
        val_run_corr = 0
        correct = 0.0
        # Eval loop
        for images, text, labels in val_loader:
            model.eval()
            images = images.to(device)
            text = text.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            torch.set_grad_enabled(False)
            outputs = model(images, text)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_run_loss += loss.item() * images.size(0)
            val_run_corr += torch.sum(preds == labels.data)
            correct += (preds == labels).sum().item()
        
        val_epoch_loss = val_run_loss / num_val_data
        val_epoch_acc = (val_run_corr.double() / num_val_data) * 100
        print(f' Epoch [{epoch + 1}/{num_epochs}] VALIDATION Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}%')

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_img_w = copy.deepcopy(model.image_classifier.state_dict())
            best_text_w = copy.deepcopy(model.text_classifier.main.state_dict())
            best_comb_w = copy.deepcopy(model.combiner.state_dict())
            print(f"New best accuracy! {best_acc:.4f}")
        print("\n")

    print(f"Best val accuracy: {best_acc:4f}")
    model.image_classifier.load_state_dict(best_img_w)
    model.text_classifier.main.load_state_dict(best_text_w)
    model.combiner.load_state_dict(best_comb_w)
    torch.save({
            'img_state_dict': model.image_classifier.state_dict(),
            'text_state_dict': model.text_classifier.main.state_dict(),
            'combiner_state_dict': model.combiner.state_dict(),
            }, 'models/combined_model.pt')


if __name__ == '__main__':
    print(device)
    train()


        