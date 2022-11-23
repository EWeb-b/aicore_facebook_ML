import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from image_processor import ImageProcessor
from text_processor import TextProcessor
from transformers import BertModel
import numpy as np

class TextClassifier(nn.Module):
    def __init__(self, input_size: int = 768, num_classes: int = 13, decoder) -> None:
        super().__init__()
        self.decoder = decoder
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
                                  nn.Linear(384 , num_classes))
        
    def forward(self, input):
        descriptions = []
        self.bert.eval()
        for stack in input:
            split = torch.unbind(stack)
            encoded = {}
            encoded['input_ids'] = split[0]
            encoded['token_type_ids'] = split[1]
            encoded['attention_mask'] = split[2]

            with torch.no_grad():
                description = self.bert(**encoded).last_hidden_state.swapaxes(1,2)
            description = description.squeeze(0) # removes the leading dimension of '1' in .size
            descriptions.append(description)

        desc_stack = torch.stack(descriptions)
        result = self.main(desc_stack)
        return result

    def predict(self, text):
        with torch.no_grad():
            prediction = self.forward(text)
            return prediction
    
    def predict_proba(self, text):
        with torch.no_grad():
            probability = torch.softmax(self.forward(text), 1)
            return probability

    def predict_classes(self, text):
        with torch.no_grad():
            res = self.forward(text)
            return self.decoder[int(torch.argmax(res, 1))]

class ImageClassifier(torch.nn.Module):
    def __init__(self, num_classes:int = 13, decoder) -> None:
        super().__init__()
        self.decoder = decoder
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, imgs):
        return self.resnet50(imgs)

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, 1)

    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, 1))]

class CombinedModel(nn.Module):
    def __init__(self, input_size: int = 768, num_classes:int = 13, decoder) -> None:
        super().__init__()
        self.decoder = decoder
        self.image_classifier = ImageClassifier()
        self.text_classifier = TextClassifier(input_size=input_size, num_classes=num_classes)
        self.combiner = nn.Linear(26, num_classes)

    def forward(self, image_features, text_features):
        img_result = self.image_classifier(image_features)
        text_result = self.text_classifier(text_features)
        combined = torch.cat((img_result, text_result), 1)
        combined_result = self.combiner(combined)
        return combined_result

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return torch.softmax(x, 1)

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            x = self.forward(image_features, text_features)
            return self.decoder[int(torch.argmax(x, 1))]


# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str

with open('aicore_facebook_ML/decoder.pkl', 'rb') as f:
    decoder = pickle.load(f)

try:
    text_classifier = TextClassifier(decoder=decoder)
    text_classifier.load_state_dict(torch.load('models/text_model.pt'), strict=False)
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
    image_classifier = ImageClassifier(decoder = decoder)
    image_classifier.load_state_dict(torch.load('models/image_model.pt'), strict=False)
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    # TODO
    combined_classifier = CombinedModel(decoder = decoder)
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
    text_processor = TextProcessor(max_length=200)
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

try:
    image_processor = ImageProcessor()
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/text')
def predict_text(text: TextItem):
  
    processed_text = text_processor(text)
    pred = text_classifier.predict(processed_text)
    prob = text_classifier.predict_proba(processed_text)
    classes = text_classifier.predict_classes(processed_text)
    print(pred)
    print(prob)
    print(classes)

    return JSONResponse(content={
        "pred": pred,
        "prob": prob,
        "classes": classes
            })
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    img_arr = np.array(pil_image)

    processed_image = image_processor(img_arr)
    pred = image_classifier.predict(processed_image)
    prob = image_classifier.predict_proba(processed_image)
    classes = image_classifier.predict_classes(processed_image)
    print(pred)
    print(prob)
    print(classes)

    return JSONResponse(content={
        "pred": pred,
        "prob": prob,
        "classes": classes
            })
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    img_arr = np.array(pil_image)
    processed_image = image_processor(img_arr)
    processed_text = text_processor(text)
    
    pred = combined_classifier.predict(processed_image, processed_text)
    prob = combined_classifier.predict_proba(processed_image, processed_text)
    classes = combined_classifier.predict_classes(processed_image, processed_text)
    print(pred)
    print(prob)
    print(classes)

    return JSONResponse(content={
        "pred": pred,
        "prob": prob,
        "classes": classes
            })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)