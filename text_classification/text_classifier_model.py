import torch
import torch.nn as nn
from transformers import BertModel

class TextClassifier(nn.Module):
    def __init__(self, input_size: int = 768, num_classes: int = 13) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.main = nn.Sequential(nn.Dropout1d(0.1),
                                  nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),

                                  nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),

                                  nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  
                                  nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Dropout1d(0.1),

                                  nn.Flatten(),
                                  nn.Dropout1d(0.3),
                                  nn.Linear(384 , 13))

    def forward(self, input):
        embeddings = []
        self.bert.eval()
        for stack in input:
            split = torch.unbind(stack)
            encoded = {}
            encoded['input_ids'] = split[0]
            encoded['token_type_ids'] = split[1]
            encoded['attention_mask'] = split[2]

            with torch.no_grad():
                embedding = self.bert(**encoded).last_hidden_state.swapaxes(1,2)
            embedding = embedding.squeeze(0) # removes the leading dimension of '1' in .size
            embeddings.append(embedding)

        desc_stack = torch.stack(embeddings)
        result = self.main(desc_stack)
        return result


        