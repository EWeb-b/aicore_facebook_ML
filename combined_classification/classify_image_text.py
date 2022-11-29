import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

from combined_classification.image_text_dataset import ImageTextDataset
from combined_classification.image_text_model import ImageTextClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_epochs = 15        
batch_size=64
lr = 0.0001
momentum=0.9
shuffle = True
num_workers=1
pin_memory=True
max_length = 200
model = ImageTextClassifier(num_classes = 13)
model = model.to(device)
optimiser = torch.optim.Adam([*model.image_classifier.parameters(), *model.text_classifier.main.parameters(), *model.combiner.parameters()], lr=lr)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

dataset = ImageTextDataset(labels_level=0, max_length=max_length, read_csv=True)
num_train_data = len(dataset) - round((len(dataset) * 0.25))
num_val_data = len(dataset) - num_train_data
train_set, validation_set = random_split(dataset,[num_train_data, num_val_data])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
val_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

title = f"imagetext_lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}, max_length={max_length}"
layout = {
    "Loss and Accuracy": {
        "loss": ["Multiline", ["loss/training", "loss/validation"]],
        "accuracy": ["Multiline", ["accuracy/training", "accuracy/validation"]],
    },
}
writer = SummaryWriter(f"drive/MyDrive/runs/imagetext_lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}, max_length={max_length}")
writer.add_custom_scalars(layout)


def train_model(mode:str, dataloader:DataLoader):
    running_loss = 0.0
    running_correct = 0
    if mode == "train":
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(mode == "train"):
        for images, text, labels in dataloader:
            images = images.to(device)
            text = text.to(device)
            labels = labels.to(device)

            if mode == "train": 
                optimiser.zero_grad()

            outputs = model(images, text)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if mode == "train":
                loss.backward()
                optimiser.step()
            
            running_loss += loss.item() * images.size(0)
            running_correct += torch.sum(preds == labels.data)
        
    return running_loss, running_correct

def calc_epoch_loss(running_loss: float, num_data: int) -> float:
    epoch_loss = running_loss / num_data
    return epoch_loss

def calc_epoch_acc(running_correct, num_data: int) -> float:
    epoch_acc = (running_correct.double() / num_data) * 100
    return epoch_acc

def write_to_summary_writer(mode, loss, accuracy, epoch):
    writer.add_scalar(f"loss/{mode}", loss, epoch)
    writer.add_scalar(f"accuracy/{mode}", accuracy, epoch)

def save_best_weights():
    best_img_w = copy.deepcopy(model.image_classifier.state_dict())
    best_text_w = copy.deepcopy(model.text_classifier.main.state_dict())
    best_comb_w = copy.deepcopy(model.combiner.state_dict())
    return best_img_w, best_text_w, best_comb_w

def main_loop():
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("-~-" * 30, f"\nEpoch [{epoch + 1}/{num_epochs}]\n", "-~-" * 30)      

        # Training loop.
        trn_run_loss, trn_run_corr = train_model("train", train_loader)
        trn_epoch_loss = calc_epoch_loss(trn_run_loss, num_train_data)
        trn_epoch_acc = calc_epoch_acc(trn_run_corr, num_train_data)
        write_to_summary_writer("training", trn_epoch_loss, trn_epoch_acc, epoch)
        print(f' Epoch [{epoch + 1}/{num_epochs}] TRAINING Loss: {trn_epoch_loss:.4f}, Acc: {trn_epoch_acc:.4f}%')

        # Eval loop.
        val_loss, val_correct = train_model("eval", val_loader)
        val_epoch_loss = calc_epoch_loss(val_loss, num_val_data)
        val_epoch_acc = calc_epoch_acc(val_correct, num_val_data)
        write_to_summary_writer("validation", val_epoch_loss, val_epoch_acc)
        print(f' Epoch [{epoch + 1}/{num_epochs}] VALIDATION Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}%')

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_img_w, best_text_w, best_comb_w = save_best_weights()
            print(f"New best accuracy! {best_acc:.4f}\n")

    print(f"Best val accuracy: {best_acc:4f}")
    model.image_classifier.load_state_dict(best_img_w)
    model.text_classifier.main.load_state_dict(best_text_w)
    model.combiner.load_state_dict(best_comb_w)
    torch.save({
            'img_state_dict': model.image_classifier.state_dict(),
            'text_state_dict': model.text_classifier.main.state_dict(),
            'combiner_state_dict': model.combiner.state_dict(),
            }, 'combined_model.pt')


if __name__ == '__main__':
    print(device)
    main_loop()

        