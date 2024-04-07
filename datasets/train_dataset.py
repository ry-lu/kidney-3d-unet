import numpy as np
from torch.utils.data import Dataset
import torch

from utils import extract_patch_from_voxel

class TrainKidney3DDataset(Dataset):
    def __init__(self, patches, masks, lowest_voxels, patch_size, norm_params, transformations=None):
        self.patches = patches
        self.masks = masks
        self.lowest_voxels = lowest_voxels
        self.patch_size = patch_size
        self.transformations = transformations
        self.std = norm_params[0]
        self.mean = norm_params[1]
        
    def __len__(self):
        return len(self.lowest_voxels)

    def __getitem__(self, idx):
        image = extract_patch_from_voxel(self.patches,self.lowest_voxels[idx],patch_size=self.patch_size)
        mask = extract_patch_from_voxel(self.masks,self.lowest_voxels[idx], patch_size=self.patch_size)
        
        image = np.expand_dims(image,0)
        mask = np.expand_dims(mask,0)
    
        data = {'image':image,'mask': mask}
        if self.transformations:
            data = self.transformations(**data)
        

        # z-score norm
        data['image']=(data['image'] - data['image'].mean()) / (data['image'].std() + 0.0001)
        data['image'] = torch.tensor(data['image'], dtype=torch.float)
        data['mask'] = torch.tensor(data['mask'], dtype=torch.float)
        return data