import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

def process_image(image: Image) -> torch.Tensor:


    image = Image.open('data/cleaned_images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg')


    transform = transforms.Compose(
            [
                transforms.ToTensor(), # This also changes the pixels to be in range [0, 1] from [0, 255].
                transforms.RandomCrop((112,112))
            ]
        )

    im_tensor = transform(image)
    im_tensor = torch.unsqueeze(im_tensor, 0) # Add the '1' dimension at the start of the tensor.

    print(im_tensor.shape)

    return im_tensor