from transformers import BertTokenizer
from transformers import BertModel

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


if __name__ == "__main__":
    text_processor = TextProcessor()
    input = text_processor("hello")
    print(input)

    print("input: ", input)
    print("type(input): ", type(input))
    descriptions = []

    #for stack in input:
    split = torch.unbind(input)
    print("split: ", split)
    print("type(split): ", type(split))
    print("split[0]: ", split[0])
    print("type(split[0]): ", type(split[0]))
    print("split[1]: ", split[1])
    print("type(split[1]): ", type(split[1]))
    print("split[2]: ", split[2])
    print("type(split[2]): ", type(split[2]))

    encoded = {}
    encoded['input_ids'] = split[0]
    encoded['token_type_ids'] = split[1]
    encoded['attention_mask'] = split[2]
    print(encoded)

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    with torch.no_grad():
        embedding = model(**encoded).last_hidden_state.swapaxes(1,2)
    #embedding = embedding.squeeze(0) # removes the leading dimension of '1' in .size

    print(embedding.shape)

