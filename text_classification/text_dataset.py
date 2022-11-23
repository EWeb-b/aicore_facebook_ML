from unicodedata import category
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import sys
# sys.path.append("..")
from ..utilities import clean_tabular_data

class TextDataset(Dataset):
    """ The TextDataset inherits its methods from the
    torch.utils.data.Dataset module.
    It starts by loading a cleaned DataFrame of text data by using the
    clean_tab_data function in clean_tabular_data.py in utilities.
    When __getitem__ is called, this class returns the text data after
    it has been tokenized by the BertTokenizer, along with the product
    category.

    Args:
        filepath: str, the path to the product data csv file.
        labels_level: int, how far back to strip the product category.
            Higher numbers result in more categories, which are more
            specific.
        max_length: int, the maximum length of the text going into the
        BERT tokenizer. Higher lengths make the model more accurate
        bu slow computation.

    Attributes:
        data: pandas.DataFrame, the text data.
        decoder: dict, used for swapping between category names and their
            integer representation.
        tokenizer: BertTokenizer, the tokenizer for processing the text.
        max_length: int, the maximum length of the text going into the
            BERT tokenizer.
    """
    def __init__(self,
                filepath: str = '/home/edwardwebb/Documents/aicore/aicore_facebook_ML/data/Products.csv',
                labels_level: int = 0,
                max_length: int = 50) -> None:
        super().__init__()
        products = clean_tabular_data.clean_tab_data(filepath, labels_level)

        self.data = products
        self.data['combined'] = products['product_name'].str.cat(products['product_description'], sep=' ')
        self.decoder = dict(enumerate(products.category.cat.categories))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
    
    def __getitem__(self, index):
        example = self.data.iloc[index]
        category = example['category']
        label_code = self.data.category.cat.categories.get_loc(category)
        label = torch.as_tensor(label_code)
        sentence = example['combined']

        encoded = self.tokenizer.batch_encode_plus([sentence],
            max_length=self.max_length, padding='max_length', truncation=True)
        inp = torch.tensor(encoded['input_ids'])
        tok = torch.tensor(encoded['token_type_ids'])
        att = torch.tensor(encoded['attention_mask'])
        stack = torch.stack((inp, tok, att))

        return stack, label

    def __len__(self):
        return len(self.data)
  

if __name__ == "__main__":
    dataset = TextDataset()

    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=1)

    for i, (data, labels) in enumerate(dataloader):
        for sentence in data:
            print(sentence)
        if i == 0:
            break

    # print(dataset.data.info())