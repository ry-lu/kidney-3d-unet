import numpy as np
from torch.utils.data import Dataset
import torch

from datasets.utils import extract_patch_from_voxel, extract_3d_voxels_for_patches, filter_empty_patches_by_voxel

class TrainKidney3DDataset(Dataset):
    def __init__(self, patches, masks, patch_size, transformations=None):
        self.patches = patches
        self.masks = masks

        self.lowest_voxels = extract_3d_voxels_for_patches(patches,patch_size=patch_size,stride=(32,32,32))
        self.lowest_voxels = filter_empty_patches_by_voxel(self.lowest_voxels,
                                                           masks,patch_size=patch_size,threshold=50)

        self.patch_size = patch_size
        self.transformations = transformations
        
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