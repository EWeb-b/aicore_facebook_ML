from transformers import BertTokenizer
import torch

class TextProcessor:
    def __init__(self, max_length: int = 50):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __call__(self, sentence):
        encoded = self.tokenizer.batch_encode_plus([sentence],
            max_length=self.max_length, padding='max_length', truncation=True)
        inp = torch.tensor(encoded['input_ids'])
        tok = torch.tensor(encoded['token_type_ids'])
        att = torch.tensor(encoded['attention_mask'])
        stack = torch.stack((inp, tok, att))

        return stack