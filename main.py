import logging
from fastapi import FastAPI, File, UploadFile, Response, Request, WebSocket, BackgroundTasks
from numpy.core.numeric import NaN
from pymongo import MongoClient
from pymongo import ReturnDocument
import pandas as pd
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
from bson import ObjectId
from bson.json_util import dumps
import uvicorn
import os
import re
import pytz
import time
import json
from fastapi.middleware.cors import CORSMiddleware
from passlib.apps import custom_app_context as pwd_context
from datetime import datetime
import uuid
from pandas import DataFrame
import pymongo
from starlette.responses import StreamingResponse
import io
from train import train_data
import random
import torch
from json import dumps
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = FastAPI()

AWS_ACCESS_KEY_ID = "AKIA3CEIQSFDTE2RKTEJ"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "us-east-1"

MONGO_DB_URL = 'mongodb://admin:Aipigeon123@ec2-100-26-194-60.compute-1.amazonaws.com:27017/admin?authSource=admin'
MONGO_DB_NAME = 'aipigeondb'
COLLECTION_INTENT_DATA = 'intentdata'


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/detect')
async def getintentresponse(request: Request):
    formdata = (await request.form())
    question = formdata['question']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    sentence = tokenize(question)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.45:
        msg = tag
    else:
        msg = '对不起，我不明白'
    return Response(content=dumps(msg), media_type="application/json")



@app.websocket("/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text('Training Started')
    await train_data()
    await websocket.send_text('Training Completed')



if __name__ == "__main__":
    #uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)



