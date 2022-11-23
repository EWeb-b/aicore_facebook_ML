import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(), # This also changes the pixels to be in range [0, 1] from [0, 255].
                # transforms.RandomResizedCrop((64,64))
                transforms.Resize((64,64)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    )

    @staticmethod
    def repeat_channel(x):
            return x.repeat(3, 1, 1)

    def __call__(self, image):
        image = self.transform(image)
        image = torch.unsqueeze(image, 0) # Add the leading '1' dimension at the start of the tensor.
        return image
