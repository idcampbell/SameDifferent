import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
    
    
#### DATASET OBJECT ####
class SameDifferentDataset(Dataset):
    def __init__(self, all_comparisons, metadata):
        # Drop double congruencies.
        inds = (metadata['sameSprite']==0).values
        metadata = metadata[inds] 
        all_comparisons = all_comparisons[inds]
        
        #nsprites = metadata.shape[0]
        #comparisons = torch.cat([all_comparisons, all_comparisons, all_comparisons])
        #sameShape = torch.tensor(metadata['sameShape'].values)
        #sameColor = torch.tensor(metadata['sameColor'].values)
        #nmatch = torch.zeros([sameColor.shape[0]])
        #ys = torch.cat([sameShape, sameColor, nmatch])
        #task_variables = torch.zeros([comparisons.shape[0], 4])
        #task_variables[:nsprites, [0, 2]] = 1 
        #task_variables[nsprites:nsprites*2, [1, 3]] = 1
        #task_variables[nsprites*2:, [0, 3]] = 1
        
        nsprites = metadata.shape[0]
        comparisons = torch.cat([all_comparisons, all_comparisons])
        sameShape = torch.tensor(metadata['sameShape'].values)
        sameColor = torch.tensor(metadata['sameColor'].values)
        ys = torch.cat([sameShape,sameColor]).float()
        task_variables = torch.zeros([comparisons.shape[0], 2])
        task_variables[:nsprites, 0] = 1  
        task_variables[nsprites:, 1] = 1
        
        # Filter out some irrelevant data.
        true_inds = torch.where(ys > 0)[0]
        false_trials = torch.where(ys < 1)[0]
        false_inds = false_trials[torch.randperm(true_inds.shape[0])]
        good_inds = torch.cat([true_inds, false_inds])
        
        # Save the relevant variables.
        self.sprites = comparisons[good_inds]
        self.metadata = pd.concat([metadata, metadata]).iloc[good_inds]
        self.task_variables = task_variables[good_inds]
        self.ys = ys[good_inds]
        
    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        return self.sprites[idx], self.ys[idx], self.task_variables[idx]