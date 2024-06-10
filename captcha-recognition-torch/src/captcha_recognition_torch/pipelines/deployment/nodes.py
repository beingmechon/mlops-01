import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from typing import List
import io
import uvicorn
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ..utils import decode_predictions, correct_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    image = transform(image)#.unsqueeze(0)
    return image

def create_app(model, idx_to_char):
    app = FastAPI()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB') #convert("P")
            input_data = preprocess_image(image)
            input_data = input_data.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                input_data = input_data.to(device)
                text_batch_logits = model(input_data)
                text_batch_pred = decode_predictions(text_batch_logits.cpu(), idx_to_char)
           
            # Prepare the dataframe for visualization
            df = pd.DataFrame(columns=['prediction'])
            df['prediction'] = text_batch_pred
            df['prediction_corrected'] = df['prediction'].apply(correct_predictions)

            # Visualize the result
            # plt.imshow(image, cmap='gray')
            # plt.title(f"Predicted Text: {df['prediction_corrected'].iloc[0]}")
            # plt.axis('off')
            # plt.show()

            return df.prediction_corrected.values.tolist()
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app

def run_app(model, idx_to_char):
    app = create_app(model, idx_to_char)
    uvicorn.run(app, host="localhost", port=8000)
