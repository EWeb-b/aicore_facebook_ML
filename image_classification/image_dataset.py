import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CNNImageDataset(Dataset):
    """ The CNNImageDataset inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all images with their labels from a pickle file
    created previously.

    Args:
        pickle_file: str, the path to the pickle file containing
            the image data.
        transform: torchvision.transforms, a transforms object 
            describing the data augmentation processes to
            apply to the images. If no transforms object is passed
            then a default transformation is applied.

    Attributes:
        data: pandas.DataFrame, the image data.
        transform: torchvision:transforms, the data augmentation
            to apply to the images.
    """
    def __init__(self, pickle_file: str, transform: transforms=None):
        super().__init__()
        pikl_data = pd.read_pickle(pickle_file)
        df = pd.DataFrame()
        df['data'] = list(zip(pikl_data.img_arr, pikl_data.category))
        self.data = df

        self.transform = transform
        if transform is None:
            transform = transforms.Compose(
                [
                transforms.ToTensor(), # This also changes the pixels to be in range [0, 1] from [0, 255].
                # transforms.RandomResizedCrop((64,64))
                transforms.Resize((64,64)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )

    def __getitem__(self, index):
        img = self.data['data'][index][0]
        img = self.transform(img)
        label = torch.tensor(self.data['data'][index][1], dtype=torch.long)
        return (img, label)

    def __len__(self):
        return len(self.data)