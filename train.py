import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from model import NeuralNet
import pymongo
from pymongo import MongoClient
from pymongo import ReturnDocument
from bson import ObjectId


X_train = []
Y_train = []
all_words = []
tags = []
xy = []

def train_data(training_id):
    client = MongoClient('mongodb://admin:Aipigeon123@ec2-54-91-210-83.compute-1.amazonaws.com:27017/admin?authSource=admin',connect = False)
    db = client['aipigeondb']
    collection = db['traininghistory']
    myquery = { "_id": ObjectId(training_id) }
    newvalues = { "$set": { "processing": True} }
    collection.update_one(myquery, newvalues)
    global tags
    global xy
    global all_words
    global X_train
    global Y_train
    X_train = []
    Y_train = []
    all_words = []
    tags = []
    xy = []
    client = MongoClient('mongodb://admin:Aipigeon123@ec2-54-91-210-83.compute-1.amazonaws.com:27017/admin?authSource=admin',connect = False)
    db = client['aipigeondb']
    collection = db['intentdata']
    data = collection.find({})
    for count,intent in enumerate(data):
        if 'intent' in intent and 'question' in intent and 'answer' in intent:
            tag = intent['answer']
            tags.append(tag)
            w = tokenize(intent['question'])
            all_words.extend(w)
            xy.append((w,tag))
        
    ignore_words = ['.',',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence,all_words)
        X_train.append(bag)
        label = tags.index(tag)
        Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)


    # Hyper-parameters 
    num_epochs = 200
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)
    print(input_size, output_size)



    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size,hidden_size,output_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    client = MongoClient('mongodb://admin:Aipigeon123@ec2-54-91-210-83.compute-1.amazonaws.com:27017/admin?authSource=admin',connect = False)
    db = client['aipigeondb']
    collection = db['traininghistory']
    myquery = { "_id": ObjectId(training_id) }
    newvalues = { "$set": { "training": True} }
    collection.update_one(myquery, newvalues)

    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        print(f'final loss: {loss.item():.4f}')

    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }

    FILE = "/home/ubuntu/ai/models/model.pth"
    client = MongoClient('mongodb://admin:Aipigeon123@ec2-54-91-210-83.compute-1.amazonaws.com:27017/admin?authSource=admin',connect = False)
    db = client['aipigeondb']
    collection = db['traininghistory']
    myquery = { "_id": ObjectId(training_id) }
    newvalues = { "$set": { "completed": True} }
    collection.update_one(myquery, newvalues)
    
    
    torch.save(data, FILE)
    print('Model Saved')

    return 'Done'

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples