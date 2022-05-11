import os 
import click 

import torch as th 
import numpy as np 
from loguru import logger 
from datasets import load_dataset 

from torch.utils.data import TensorDataset, DataLoader
from model  import MLP_Model

from rich.console import Console 

from tqdm import tqdm 
from libraries.strategies import * 

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=False)
@click.pass_context
def router_cmd(ctx, debug):
    ctx.obj['debug_mode'] = debug 
    invoked_subcommand = ctx.invoked_subcommand 
    if invoked_subcommand is None:
        logger.debug('no subcommand were called')
    else:
        logger.debug(f'{invoked_subcommand} was called')


@router_cmd.command()
@click.option('--dataset_name', help='name of the dataset to download', type=str)
@click.option('--path2source_data', help='path where the dataset will be saved', type=click.Path(False))
def grabbing(dataset_name, path2source_data):
    logger.debug('dataset downloading')
    data = load_dataset(dataset_name, split=['train', 'validation', 'test'])
    serialize(path2source_data, data)  # store dataset into source_data

@router_cmd.command()
@click.option('--path2source_data', help='path to dataset', type=click.Path(True))
@click.option('--path2models', help='models location', type=click.Path(True))
@click.option('--model_name', help='name of the model', type=str)
@click.option('--path2features', help='path to features', type=click.Path(False))
@click.option('--nb_limit', type=int, help='dataset size limit')
def embedding(path2source_data, path2models, model_name, path2features, nb_limit):
    data = deserialize(path2source_data)
    train, valid, evalu = data 

    logger.debug(f'nb item in train : {len(train)}')
    logger.debug(f'nb item in valid : {len(valid)}')
    logger.debug(f'nb item in evalu : {len(evalu)}')
    
    vectorizer = load_model(path2models, model_name)

    labels_acc = []
    vectors_acc = []
    counter = 0
    for item in tqdm(train):
        review = item['review']
        fingerprint = vectorizer.encode(review).tolist()
        vectors_acc.append(fingerprint)
        label = item['label']
        labels_acc.append(label)
        counter += 1 
        if counter == nb_limit:
            break 

    zipped_vectors_labels = list(zip(vectors_acc, labels_acc))
    serialize(path2features, zipped_vectors_labels)
    logger.success('vectorization successfull')

@router_cmd.command()
@click.option('--path2features', help='path to features', type=click.Path(True))
@click.option('--nb_epochs', type=int, help='number of epochs')
@click.option('--bt_size', help='batch size', type=int)
def learning(path2features, nb_epochs, bt_size):
    logger.debug('load features')
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    features = deserialize(path2features)
    
    vectors, labels = list(zip(*features))
    stacked_vectors = th.tensor(np.vstack(vectors)).float()
    stacked_labels = th.tensor(np.vstack(labels)).float()

    # create dataset 
    dataset = TensorDataset(stacked_vectors, stacked_labels)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=bt_size)

    net = MLP_Model(layer_cfg=[512, 256, 64, 1],  non_linears=[1, 1, 0], dropouts=[0.1, 0.1, 0.0])
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    criterion = th.nn.BCEWithLogitsLoss()  # binary classification : postive or negative 

    net.to(device) # move model to cuda device 
    print(net)
    nb_data = len(dataset)

    for epoch in range(nb_epochs):
        counter = 0
        for X, Y in tqdm(data_loader): 
            counter += len(X)

            X = X.to(device)  # vectors
            Y = Y.to(device)  # labels 

            P = net(X)

            optimizer.zero_grad()
            E = criterion(P, Y)
            E.backward()
            optimizer.step()

        logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] [{counter:05d}/{nb_data:05d}] >> Loss : {E.cpu().item():07.3f}')

    th.save(net.cpu(), 'network.th')
    logger.success('the model was saved ...!')


@router_cmd.command()
@click.option('--path2models', help='path to models', type=click.Path(True))
@click.option('--model_name', help='vectorization', type=str)
@click.option('--path2network', help='path to predictor', type=click.Path(True))
def inference(path2models, model_name, path2network):
    logger.debug('load vectorizer (sentence transformer)')
    vectorizer = load_model(path2models, model_name)
    net = th.load(path2network)
    console = Console()

    keep_prediction = True 
    while keep_prediction:
        movie_review = input('give a review: ')
        if movie_review == 'stop':
            keep_prediction = False
        else:
            fingerprint = vectorizer.encode(movie_review)
            fingerprint = th.tensor(fingerprint).float()
            probability = net(fingerprint[None, ...]).squeeze(0).sigmoid().item()  # postive| negative 
            
            logger.debug(movie_review)
            logger.debug(f'probability : {probability:07.3f}')
            emotion = ':grinning_face:' if probability > 0.6 else ':angry_face:' if probability < 0.4 else ':neutral_face:'
            console.print(f'predicted emotion : {emotion}')        


if __name__ == '__main__':
    router_cmd(obj={})