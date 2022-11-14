from transformers import BertTokenizer
from transformers import BertModel
import torch


def process_sentence(sentence: str, max_length: int):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    encoded = tokenizer.batch_encode_plus([sentence], max_length=max_length, padding='max_length', truncation=True)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}

    with torch.no_grad():
        result = model(**encoded).last_hidden_state.swapaxes(1,2)

    return result