import torch
import torch.nn.functional as F

class ImageTransferCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Linear(num_ftrs, 13)

    def forward(self, imgs):
        return self.resnet50(imgs)
