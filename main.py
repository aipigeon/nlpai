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
from train import train_data , train_data_game
import random
import torch
from json import dumps
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from typing import Dict
import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from http import HTTPStatus
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


app = FastAPI()

AWS_ACCESS_KEY_ID = "AKIA3CEIQSFDTE2RKTEJ"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "us-east-1"

MONGO_DB_URL = 'mongodb://admin:Aipigeon123@ec2-54-91-210-83.compute-1.amazonaws.com:27017/admin?authSource=admin'
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




class Job(BaseModel):
    uid: UUID = Field(default_factory=uuid4)
    status: str = "in_progress"
    result: int = None

jobs: Dict[UUID, Job] = {}



# async def run_in_process(fn, *args):
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(app.state.executor, fn, *args)  # wait and return result


# async def start_cpu_bound_task(trainingid: str, game: str) -> None:
#     jobs[uid].result = await run_in_process(train_data_game, game, trainingid)
#     jobs[uid].status = "complete"



# @app.on_event("startup")
# async def startup_event():
#     app.state.executor = ProcessPoolExecutor()


# @app.on_event("shutdown")
# async def on_shutdown():
#     app.state.executor.shutdown()

@app.post('/detect/{game}')
async def getintentresponse(game,request: Request):
    formdata = (await request.form())
    question = formdata['question']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = game + '.pth'
    FILE = "models/" + filename
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
    if prob.item() > 0.50:
        msg = tag
    else:
        msg = '对不起，我不明白'
    return Response(content=dumps(msg), media_type="application/json")


# @app.post('/detect')
# async def getintentresponse(request: Request):
#     formdata = (await request.form())
#     question = formdata['question']
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     FILE = "models/model.pth"
#     data = torch.load(FILE)

#     input_size = data["input_size"]
#     hidden_size = data["hidden_size"]
#     output_size = data["output_size"]
#     all_words = data['all_words']
#     tags = data['tags']
#     model_state = data["model_state"]

#     model = NeuralNet(input_size, hidden_size, output_size).to(device)
#     model.load_state_dict(model_state)
#     model.eval()
#     sentence = tokenize(question)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.45:
#         msg = tag
#     else:
#         msg = '对不起，我不明白'
#     return Response(content=dumps(msg), media_type="application/json")



# @app.websocket("/train")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     await websocket.send_text('Training Started')
#     await train_data()
#     await websocket.send_text('Training Completed')



# @app.post('/data/train')
# async def traindata(request: Request, background_tasks: BackgroundTasks):
#     client = MongoClient('mongodb://admin:Aipigeon123@ec2-54-91-210-83.compute-1.amazonaws.com:27017/admin?authSource=admin',connect = False)
#     db = client['aipigeondb']
#     collection = db['traininghistory']
#     formdata = (await request.form())
#     userid = formdata['userid']
#     ipaddress = formdata['ipaddress']
#     lasthistory = collection.find({}).sort('_id',-1).limit(1)
#     if 'completed' in lasthistory[0]:
#         mydict = { "request_recieved": True, "userid": userid,"ipaddress":ipaddress,"date": datetime.now(pytz.timezone('Asia/Dubai'))}
#         collection.insert_one(mydict)
#         new_task = Job()
#         jobs[new_task.uid] = new_task
#         background_tasks.add_task( start_cpu_bound_task ,mydict['_id'],new_task.uid)
#         return Response(content=dumps('Training Requested'), media_type="application/json")
#     else:
#         return Response(content=dumps('Training In Progress'), media_type="application/json")




@app.post('/data/train/{game}')
async def gameagenttraindata(game,request: Request, background_tasks: BackgroundTasks):
    client = MongoClient('mongodb://admin:Aipigeon123@ec2-54-91-210-83.compute-1.amazonaws.com:27017/admin?authSource=admin',connect = False)
    db = client['aipigeondb']
    collection = db['traininghistory']
    formdata = (await request.form())
    userid = formdata['userid']
    ipaddress = formdata['ipaddress']
    lasthistory = collection.find({}).sort('_id',-1).limit(1)
    if 'completed' in lasthistory[0]:
        mydict = { "request_recieved": True, "userid": userid,"ipaddress":ipaddress,"date": datetime.now(pytz.timezone('Asia/Dubai'))}
        collection.insert_one(mydict)
        # await train_data_game(game,mydict['_id'])
        background_tasks.add_task( train_data_game ,game,mydict['_id'])
        return Response(content=dumps({'msg':'Training Requested'}), media_type="application/json")
    else:
        return Response(content=dumps({'msg':'Training In Progress'}), media_type="application/json")
        


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



