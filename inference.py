import fastapi
import pandas as pd
from fastapi import File, UploadFile
from feature_extraction import mfcc_extraction
from fastapi.responses import FileResponse
from hmm_model import load_model
from common import LABELS
import numpy as np
LABELS.remove("sil")
def predict(path):
    mfcc = np.array(mfcc_extraction(path)).T
    models = load_model()
    scores = [models[label].score(mfcc) for label in LABELS]
    pred = np.argmax(scores)
    return LABELS[pred]
app = fastapi.FastAPI()

@app.post(path="/demo")
async def main(file: UploadFile = File(media_type='multipart', default='Any')):
    file_received = "inference.wav"
    with open(file_received, "wb") as file_object:
        file_object.write(file.file.read())
    pred = predict(file_received)
    return pred
