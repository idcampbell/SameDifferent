import os
import argparse

import numpy as np
import pandas as pd
import torch

from generate_data import generate_sprites, generate_features
from utils import train, test, setup


def train_model(args, split, encoder):
    '''Train the model for a set number of epochs.
    Args:
        args (list): List of parameter of the analysis.
        split (int): Subset of the data to train the model on.
        encoder (str): Encoder type of the model (conv or linear).
    '''
    model, train_loader, test_loader, train_dataset, test_dataset, optimizer, device = setup(f'data/{args.name}.pt', f'data/{args.name}_metadata.csv', train_cond='isTrain_split'+str(split)+'_all', test_cond='isTest_split'+str(split)+'_all', encoder=encoder)
    
    # Track the test and training loss of the networks during training.
    nepochs = int(args.nEpochs)
    train_losses = torch.zeros([nepochs])
    test_losses = torch.zeros([int(nepochs/10)])
    test_epochs = torch.zeros([int(nepochs/10)])
    
    print(len(train_dataset))
    print(len(test_dataset))
    
    # Train the network for nepochs.
    for i_epoch in range(nepochs):
        train_loss = train(model, device, train_loader, optimizer)
        if i_epoch % 10==0:
            test_loss = test(model, device, test_loader)
            test_losses[int(i_epoch/10)] = test_loss
            test_epochs[int(i_epoch/10)] = i_epoch / len(train_dataset)
        if i_epoch % int(args.logInterval)==0:
            torch.save(model.state_dict(), f'output/{args.name}/split{split}_epoch{i_epoch}.pt')
            print(f'Epoch {i_epoch} train set average loss: {train_loss:.8f}')
            print(f'Epoch {i_epoch} test set average loss: {test_loss:.8f} \n')
        train_losses[i_epoch] = train_loss
    return model, train_losses, test_losses, test_epochs

    
def parse_args():
    '''Parse arguments for user input when training the model.
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_type', choices=['one-hot', 'image'], help='Type of dataset to generate.')
    ap.add_argument('-n', '--name', type=str, required=True, help='Name of the dataset and metadata to load.')
    ap.add_argument('-s', '--splits', nargs='*', required=True, default=np.arange(2, 8), help='Split factors to train on.')
    ap.add_argument('-d', '--device', default='cpu', required=True, help='Device to use when training and testing the model.')
    ap.add_argument('-i', '--logInterval', default=100, help='Interval at which to print the training/testing loss.')
    ap.add_argument('-e', '--nEpochs', default=1000, help='Number of epochs to train the model.')
    args = ap.parse_args()
    return args


# Generate the full dataset according to the input parameters.
if __name__ == '__main__':
    
    # Parse the user arguments.
    args = parse_args()
    
    # Get the model encoder type.
    if args.dataset_type=='one-hot':
        encoder = 'linear'
    elif args.dataset_type=='image':
        encoder = 'conv'
    else:
        raise ValueError(f'Illegal dataset type: {args.dataset_type}')
    
    # Train a model for each of the split factors.
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    for split in args.splits:
        
        # Make sure the output directories are created.
        os.makedirs(f'output/{args.name}', exist_ok=True)
        os.makedirs(f'checkpoints/{args.name}', exist_ok=True)
        
        # Train and test the model.
        model, train_losses, test_losses, test_epochs = train_model(args, split, encoder)
        
        # Add the test and train performance.
        test_perf = torch.stack([test_losses, test_epochs])
        torch.save(test_perf, f'output/{args.name}/split{split}.pt')