import pickle as pk 

from os import path 
from sentence_transformers import SentenceTransformer as ST 

def serialize(path2dump, data):
    with open(path2dump, mode='wb') as fp:
        pk.dump(data, fp)

def deserialize(path2dump):
    with open(path2dump, mode='rb') as fp:
        return pk.load(fp)

def load_model(models_path, model_name):
    path2model = path.join(models_path, model_name)
    if path.isfile(path2model):
        model = deserialize(path2model)
        return model 
    else:
        model = ST(model_name)
        serialize(path2model, model)
        return model 
