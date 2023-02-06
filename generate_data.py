import argparse
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from webcolors import name_to_rgb
from einops import repeat


def generate_features(ncolors, nshapes):
    '''Generate the raw features for the one-hot dataset.

    Args:
        ncolors (float): Number of one-hot colors to include in the dataset.
        nshapes (float): Number of one-hot shapes to include in the dataset.
    '''
    shapes = torch.eye(nshapes)
    colors = torch.eye(ncolors)
    all_colors = torch.repeat_interleave(colors, nshapes, 0)
    all_shapes = torch.tile(shapes, (ncolors,1))
    all_sprites = torch.hstack([all_colors,all_shapes])
    color_ids = np.argwhere(all_colors>0)[1].numpy()
    shape_ids = np.argwhere(all_shapes>0)[1].numpy()
    return all_sprites, color_ids, shape_ids


def color_from_cmap(cmap, ncolors):
    '''Generate a list of RGB values from a given cmap.
    
    Args:
        cmap (str): Name of the matplotlib cmap to use when generating the colors.
        ncolor (int): Number of colors to evenly sample from the colormap.
    '''
    cmap = cm.get_cmap(cmap, ncolors)
    rgb = cmap(np.linspace(0, 1, ncolors))
    return rgb[:, :3]*255


def generate_sprites(sprites_file, ncolors, nshapes, colors_list=None, cmap=None):
    '''Generate the individual sprites for the image dataset. 
    
    Args:
        sprites_file (string): Path of the npy images of the black and white sprites.
        colors_list: (list, optional): List of the CSS2 colors to make the sprites.
        nshapes (float): Number of unique shapes to include in the dataset (max=100).
    '''
    # Generate an RGB colorlist.
    if cmap:
        rgb_colors = color_from_cmap(cmap, ncolors)
        colors_list = ['R'+str(round(c[0], 3))+'G'+str(round(c[1], 3))+'B'+str(round(c[2], 3)) for c in rgb_colors]
    else:
        rgb_colors = [name_to_rgb(color) for color in colors_list]
    sprites = torch.tensor(np.load(sprites_file))[torch.randperm(nshapes)]
    all_sprites = torch.zeros([nshapes*ncolors, 3, 32, 32])
    for i, color in enumerate(rgb_colors):
        colors_matrix = torch.ones([nshapes, 3, 32, 32])
        rgb = torch.tensor(color)
        all_color = torch.einsum('ijkl,j->ijkl', colors_matrix, rgb) / 255
        colored_shapes = torch.einsum('ijkl,ikl->ijkl', all_color, sprites) / 255
        all_sprites[i*nshapes:(i+1)*nshapes] = colored_shapes
    all_colors = np.repeat(colors_list, nshapes, 0)
    all_shapes = np.tile(np.arange(nshapes), ncolors)
    return all_sprites, all_colors, all_shapes


def split_dataset(ncolors, nshapes, split_factor=2):
    '''Generate test and train indices that split the dataset evenly over sprites.
    
    Args:
        ncolors (float): Number of colors represented in the dataset.
        nshapes: (float): Number of shapes represented in the dataset.
        split_factor: (int, optional): Number of times each sprite should appear in the SameDifferent dataset (ie. each sprite represented in split_factor comparisons)
    '''
    # Generate the correct indices to split the data in a way that preserves even feature sampling.
    nums = np.arange(ncolors)
    inds = np.zeros([ncolors, nshapes]).astype(int)
    for i in range(split_factor):
        rolled_inds = np.roll(nums, -i)
        inds[rolled_inds, np.arange(nshapes)] = 1
    # Now return the training and testing set indices.
    train_inds = np.ravel(inds).astype(bool)
    test_inds = np.logical_not(train_inds)
    return train_inds, test_inds


def create_comparisons(sprites, colors, shapes):
    '''Generate all possible comparisons for the SameDifferent dataset (works with either image or one-hot datasets).
    
    Args:
        sprites (tensor): Tensor of all sprites (either one-hot sprites (nbatch, nfeatures) or image sprites (nbatch, 3, 32, 32)).
        colors: (np.array): Array of sprite colors.
        shapes (np.array): Array of sprite shape IDs.
    '''
    n = sprites.shape[0]
    comparisons = torch.zeros([n**2, 2, *sprites[0].shape])
    metadata = pd.DataFrame(np.zeros([n**2, 10]), columns=['shape1', 'color1', 'shape2', 'color2', 'ID1', 'ID2', 'compID', 'sameShape', 'sameColor', 'sameSprite'])
    for i, sprite in enumerate(sprites):
        # Add shape, color, and ID fields to metadata.
        sprite1ID =  np.char.array(np.repeat(colors[i], n)) + '-' + np.char.array(np.repeat(shapes[i], n).astype(str))
        sprite2ID = np.char.array(colors) + '-' + np.char.array(shapes.astype(str)) 
        metadata.iloc[i*n:(i+1)*n, :7] = np.array([shapes[i], colors[i], shapes, colors, sprite1ID, sprite2ID, sprite1ID+'/'+sprite2ID], dtype='object')
        # Add the images to the comparisons tensor.
        comparisons[i*n:(i+1)*n, 0] = torch.stack([sprite for _ in range(n)])
        comparisons[i*n:(i+1)*n, 1] = sprites
    metadata['sameShape'] = (metadata['shape1'].values==metadata['shape2'].values).astype(int)
    metadata['sameColor'] = (metadata['color1'].values==metadata['color2'].values).astype(int)
    metadata['sameSprite'] = (metadata['sameShape'].values & metadata['sameColor'].values).astype(int)
    return comparisons, metadata


def get_split_inds(metadata, split_factor=2):
    '''Compute indices of non-match stimulus pairs to include in the dataset.
    
    Args:
        metadata (pd.DataFrame): Dataframe including information about each stimulus pair. 
        split_factor: (int, optional): Number of stimuli to sample per unique feature value.
    '''
    match_mask = (metadata['sameColor']==1).values | (metadata['sameShape']==1).values
    nonmatch_mask = np.zeros_like(match_mask)
    for spriteID in np.unique(train_metadata.ID1.values):
        sprite_mask = train_metadata.ID1.values==spriteID
        nonmatch_inds = np.where(sprite_mask & ~match_mask)[0]
        np.random.shuffle(nonmatch_inds)
        nonmatch_mask[nonmatch_inds[:split_factor+1]] = 1 # Grab the first n non-matches.
    return metadata[match_mask | nonmatch_mask].compID.values


def parse_args():
    '''Parse arguments for user input when generating the dataset.
    '''
    color_ids = ['aqua', 'blue', 'fuchsia', 'green', 'grey', 'lime', 'maroon', 'navy', 'olive', 'purple', 'red', 'silver', 'teal', 'white', 'yellow', 'orange']
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_type', choices=['one-hot', 'image'], help='Type of dataset to generate.')
    ap.add_argument('-n', '--name', type=str, required=True, help='Name to use when saving the dataset and metadata.')
    ap.add_argument('-s', '--nShapes', type=str, default=16, required='True', help='Number of shapes to include in the datasets.')
    ap.add_argument('-c', '--nColors', type=str, default=16, required='True', help='Number of colors to include in the datasets.')
    ap.add_argument('-p', '--path', help='Path of sprites data to generate the dataset.')
    ap.add_argument('-l', '--colors', nargs='*', default=color_ids, help='Colors to use when generating the dataset.')
    ap.add_argument('-i', '--splits', nargs='*', default=np.arange(2, 8), help='Split factors to use when generating the dataset.')
    ap.add_argument('-m', '--cmap', default=None, help='Color map to use when generating the data. If provided, overrides the input colorlist.')
    args = ap.parse_args()
    return args


# Generate the full dataset according to the input parameters.
if __name__ == '__main__':
    
    # Parse the user arguments.
    args = parse_args()
    nColors = int(args.nColors)
    nShapes = int(args.nShapes)
    
    # Generate the sprites.
    if args.dataset_type=='one-hot':
        sprites, colors, shapes = generate_features(nColors, nShapes)
    elif args.dataset_type=='image':
        sprites, colors, shapes = generate_sprites(args.path, nColors, nShapes, cmap=args.cmap, colors_list=args.colors)
    else:
        raise ValueError(f'Illegal dataset type: {args.dataset_type}')
        
    # Generate all comparisons among the items and save them to a tensor object.
    all_comparisons, all_metadata = create_comparisons(sprites, colors, shapes)
    os.makedirs('data', exist_ok=True)
    torch.save(sprites, f'data/{args.name}_sprites.pt')
    torch.save(all_comparisons, f'data/{args.name}.pt')
    
    # Now generate the dataset metadata with the relevant dataset split indices.
    for split in args.splits:
        
        # Compute the indices to subsample the dataset according to the current split factor.
        train_inds, holdout_inds = split_dataset(nColors, nShapes, split_factor=int(split))
        
        # Generate all comparisons of the valid subset.
        train_comparisons, train_metadata = create_comparisons(sprites[train_inds], colors[train_inds], shapes[train_inds])
        test_comparisons, test_metadata = create_comparisons(sprites[holdout_inds], colors[holdout_inds], shapes[holdout_inds])
        #all_train_metadata = all_train_metadata[all_train_metadata['sameSprite']==0] # Drop double congruencies.
        all_metadata['isTrain_split'+str(split)+'_all'] = all_metadata['compID'].isin(train_metadata.compID.values).values
        all_metadata['isTest_split'+str(split)+'_all'] = all_metadata['compID'].isin(test_metadata.compID.values).values
    
        # Limit training examples by subsampling the non-matches.
        #train_comp_ids = get_split_inds(train_metadata, split_factor=int(split))
        #all_metadata['isTrain_'+'split'+str(split)+'_reduced'] = all_metadata['compID'].isin(train_comp_ids).values

    # Save the fina metadata.
    all_metadata.to_csv(f'data/{args.name}_metadata.csv')
