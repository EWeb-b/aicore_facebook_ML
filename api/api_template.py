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
##############################################################
# TODO                                                       #
# Import your image and text processors here                 #
##############################################################


class TextClassifier(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(TextClassifier, self).__init__()
        pass

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the text model    #
##############################################################
        
        self.decoder = decoder
    def forward(self, text):
        x = self.main(text)
        return x

    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x
    
    def predict_proba(self, text):
        with torch.no_grad():
            pass


    def predict_classes(self, text):
        with torch.no_grad():
            pass

class ImageClassifier(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the image model   #
##############################################################
        
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            pass

    def predict_classes(self, image):
        with torch.no_grad():
            pass


class CombinedModel(nn.Module):
    def __init__(self,
                 decoder: list = None):
        super(CombinedModel, self).__init__()
##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the combined model#
##############################################################
        
        self.decoder = decoder

    def forward(self, image_features, text_features):
        pass

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            pass

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            pass



# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the text model    #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# text_decoder.pkl                                           #
##############################################################
    pass
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the image model   #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# image_decoder.pkl                                          #
##############################################################
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the combined model#
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# combined_decoder.pkl                                       #
##############################################################
    pass
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the text processor that you will use to process #
# the text that you users will send to your API.             #
# Make sure that the max_length you use is the same you used #
# when you trained the model. If you used two different      #
# lengths for the Text and the Combined model, initialize two#
# text processors, one for each model                        #
##############################################################
    pass
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################
    pass
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
  
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the text model   #
    # text.text is the text that the user sent to your API       #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
        "Category": "", # Return the category here
        "Probabilities": "" # Return a list or dict of probabilities here
            })
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # In this case, text is the text that the user sent to your  #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)