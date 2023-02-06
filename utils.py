import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import SameDifferentDataset
from models import CoRelNet


def setup(datapath, metadata_path, encoder='linear', train_cond='isTrain_split2_all', test_cond='isTest_split2_all', seed=88, use_gpu=True, batch_size=256, lr=1e-3):
    '''Setup the training environment.
    
    Args:
        datapath (str): Path to the raw training data.
        metadata_path (str): Path to the metadata.
        encoder (str, optional): Type of encoder to use for the CoRelNet.
        train_cond (str, optional): Name of the training dataset mask to use.
        seed (int, optional): Seed to set the training model.
        use_gpu (boolean, optional): Whether or not to train using the GPU.
        batch_size (int, optional): Batch size for model fitting. 
        lr (int, optional): Learning rate for model fitting.
    '''
    # Set up the analysis.
    use_cuda = use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('using device: '+str(device))
    # Set up the model that we want to use.
    model = CoRelNet(encoder=encoder).to(device)
    # Set up the datasets.
    all_comparisons = torch.load(datapath)
    metadata = pd.read_csv(metadata_path)
    #train_mask = metadata[train_cond].values
    #holdout_mask = metadata[test_cond].values
    train_mask = np.zeros([metadata.shape[0]]).astype(bool)
    holdout_mask = np.zeros_like(train_mask).astype(bool)
    train_mask[torch.randperm(int(train_mask.shape[0]*0.5))] = 1
    holdout_mask[~train_mask] = 1
    #holdout_mask = ~mask[torch.randperm(mask.sum())]
    #train_comps, train_metadata, test_comps, test_metadata = all_comparisons[train_mask], metadata[train_mask], all_comparisons[holdout_mask], metadata[holdout_mask]
    train_comps, train_metadata, test_comps, test_metadata = all_comparisons[train_mask], metadata.iloc[train_mask], all_comparisons[holdout_mask], metadata.iloc[holdout_mask]
    train_dataset = SameDifferentDataset(train_comps, train_metadata)
    test_dataset = SameDifferentDataset(test_comps, test_metadata)
    # Create dataloaders for training and testing.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # Create optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, train_loader, test_loader, train_dataset, test_dataset, optimizer, device


def generate_mphate_examples(sprites, device, nx=4, nz=16, nepochs=1000):
    '''Generate the variables to store the model encodings during training.
    
    Args:
        sprites (torch.Tensor): sprites to compute the representations for.
        device (str): device to test the model on (CPU or CUDA).
        nx (int, optional): size of the comparison layer representations.
        nz (int, optional): size of the in-context layer representations.
        nepochs (int, optional): number of epochs to store the latent representations.
    
    '''
    nitems = sprites.shape[0]
    task1 = torch.zeros([nitems, 2]).to(device)
    task2 = torch.zeros([nitems, 2]).to(device)
    task1[:, 0] = 1
    task2[:, 1] = 1
    xs = torch.zeros([nepochs, nitems*2, 4]).to(device)
    zs = torch.zeros([nepochs, nitems*2, 16]).to(device)
    return sprites.to(device), xs, zs, task1, task2 


def train(model, device, train_loader, optimizer):
    ''' Function to manage the training of the network.
    
    Args:
        model (nn.Module): model object to train.
        device (str): device to train the model on (CPU or CUDA).
        train_loader (torch.utils.data.DataLoader): dataloader object for the training data.
        optimizer (torch.optim): optimizer object to train the network weights.
    '''
    model.train()
    train_loss = 0
    for batch_idx, (sprites, ys, tasks) in enumerate(train_loader):
        optimizer.zero_grad()
        sprites, ys, tasks = sprites.to(device), ys.to(device), tasks.to(device)
        output, x1, x2, z1, z2 = model(sprites[:,0], sprites[:,1], tasks)
        loss = model.loss(output, ys)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    return train_loss

# Function to manage the testing of the network.
def test(model, device, test_loader):
    '''Function to manage the testing of the network.
    
    Args:
        model (nn.Module): model object to test.
        device (str): device to test the model on (CPU or CUDA).
        test_loader (torch.utils.data.DataLoader): dataloader object for the testing data. 
    '''
    model.eval()
    test_loss = 0
    with torch.no_grad():
        # Now evaluate the model on the real test dataset.
        for batch_idx, (sprites, ys, tasks) in enumerate(test_loader):
            sprites, ys, tasks = sprites.to(device), ys.to(device), tasks.to(device)
            output, x1, x2, z1, z2 = model(sprites[:,0], sprites[:,1], tasks)
            loss = model.loss(output, ys)
            test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
    return test_loss



'''
# Function to manage the testing of the network.
def test(model, device, test_loader, test_sprites, task1, task2, xs, zs, epoch, log_interval=1000):
Function to manage the testing of the network.
    
    Args:
        model (nn.Module): model object to test.
        device (str): device to test the model on (CPU or CUDA).
        test_loader (torch.utils.data.DataLoader): dataloader object for the testing data. 
        test_sprites (torch.Tensor): sprites to compute the representations for.
        task1 (torch.Tensor): task condition variables for task1.
        task2 (torch.Tensor): task condition variables for task2.
        xs (torch.Tensor): Tensor to store the comparison layer encodings during testing.
        zs (torch.Tensor): Tensor to store the in-context layer encodings during testing.
        epoch (int): Number of the current epoch.
        log_interval (int, optional): interval at which to compute M-phate embeddings.
    model.eval()
    test_loss = 0
    with torch.no_grad():
        if epoch%log_interval==0:
            # Collect some data for M-PHATE plotting.
            _, x1, _, z1, _ = model(test_sprites, test_sprites, task1)
            _, x2, _, z2, _ = model(test_sprites, test_sprites, task2)
            xs[epoch, :256, :] = x1.to('cpu')
            zs[epoch, :256, :] = z1.to('cpu')
            xs[epoch, 256:, :] = x2.to('cpu')
            zs[epoch, 256:, :] = z2.to('cpu')
        # Now evaluate the model on the real test dataset.
        for batch_idx, data in enumerate(test_loader):
            sprites, ys, tasks = data
            sprites, ys, tasks = sprites.to(device), ys.to(device), tasks.to(device)
            output, x1, x2, z1, z2 = model(sprites[:,0], sprites[:,1], tasks)
            loss = model.loss(output, ys)
            test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
    return test_loss, xs, zs
'''

