import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import BertTokenizer
from torch.utils.data import Dataset
from torchvision import transforms


class ImageTextDataset(Dataset):
    def __init__(self,
                 labels_level: int = 0,
                 transform: transforms = None,
                 max_length: int = 50,
                 read_csv: bool = True) -> None:
        
        if read_csv:
            self.data = pd.read_pickle('data/merged_data.zip')
        else:
            self.data = self.fetch_clean_data(labels_level=labels_level)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                    transforms.ToTensor(), # This also changes the pixels to be in range [0, 1] from [0, 255].
                    transforms.RandomCrop((112,112))
                    # , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        row = self.data.iloc[index]
        category = row['category']
        label_code = self.data.category.cat.categories.get_loc(category)
        label = torch.as_tensor(label_code)

        sentence = row['product_description']
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        inp = torch.tensor(encoded['input_ids'])
        tok = torch.tensor(encoded['token_type_ids'])
        att = torch.tensor(encoded['attention_mask'])
        text = torch.stack((inp, tok, att))

        image = row['img']
        image = self.transform(image)

        return image, text, label

    def __len__(self):
        return len(self.data)
    

    @staticmethod
    def fetch_clean_data(labels_level):
        products = pd.read_csv('data/Products.csv', lineterminator="\n")
        images = pd.read_csv('data/Images.csv')
        merged = pd.merge(products, images, left_on='id', right_on='product_id')
        merged.drop(['id_x', 'product_name', 'price', 'location', 'product_id', 'Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)
        merged = merged.rename({'id_y': 'img'}, axis = 1)
        merged['category'] = merged['category'].apply(lambda x: x.split("/")[labels_level].strip())
        merged['category'] = merged['category'].astype('category')
        merged['img'] = merged['img'].apply(lambda z: np.array(Image.open('data/cleaned_images/' + z + '.jpg')))
        return merged


if __name__ == "__main__":
    dataset = ImageTextDataset(read_csv=False)
    print(dataset.data.info())