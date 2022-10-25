import pandas as pd
import torch

from torch.utils.data import Dataset


class CNNImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        super().__init__()
        self.data = pd.read_pickle(csv_file)
        self.transform = transform
        

    def __getitem__(self, index):
        example = self.data.iloc[index]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #feature = torch.from_numpy(example['img_arr'].transpose((2, 0, 1)))
        img = example['img_arr']
        label = torch.tensor(example['category'], dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return len(self.data) 