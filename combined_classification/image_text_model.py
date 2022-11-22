import torch
import torch.nn as nn

from text_classification.text_classifier_model import TextClassifier
from image_classification.image_transfer_CNN_model import ImageTransferCNN

class ImageTextClassifier(nn.Module):
    def __init__(self, input_size: int = 768, num_classes:int = 13) -> None:
        super().__init__()
        self.image_classifier = ImageTransferCNN()
        self.text_classifier = TextClassifier(input_size=input_size, num_classes=num_classes)
        self.combiner = nn.Linear(26, num_classes)

    def forward(self, image_features, text_features):
        img_result = self.image_classifier(image_features)
        text_result = self.text_classifier(text_features)
        combined = torch.cat((img_result, text_result), 1)
        combined_result = self.combiner(combined)
        return combined_result