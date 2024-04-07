from tqdm import tqdm
import numpy as np

def extract_3d_voxels_for_patches(image_3d, patch_size=(128, 128, 128), stride=(64, 64, 64)):
    """
    Extracts patches in an image and returns a list of lowest voxels.
    """
    patches = []

    for start_x in range(0, image_3d.shape[0] - patch_size[0], stride[0]):
        for start_y in range(0, image_3d.shape[1] - patch_size[1], stride[1]):
            for start_z in range(0, image_3d.shape[2] - patch_size[2], stride[2]):
                # Lowest voxel needed to extract patch
                lowest_voxel = (start_x,start_y,start_z)
                patches.append(lowest_voxel)
                
    return patches

def extract_patch_from_voxel(image_3d, lowest_voxel, patch_size=(128, 128, 128)):
    """
    Extracts a 3d patch given the lowest voxel.
    """
    end_x = lowest_voxel[0] + patch_size[0] 
    end_y = lowest_voxel[1] + patch_size[1]
    end_z = lowest_voxel[2] + patch_size[2]

    patch = image_3d[lowest_voxel[0]:end_x,
                     lowest_voxel[1]:end_y,
                     lowest_voxel[2]:end_z]

    return patch



def filter_empty_patches_by_voxel(lowest_voxels, mask, patch_size, threshold = 0):
    """
    Removes the lowest voxels that represent patches with positive values <= threshold
    """
    positive_voxels = []

    for lowest_voxel in tqdm(lowest_voxels):
        # Extract the patch
        patch = mask[lowest_voxel[0]:lowest_voxel[0]  + patch_size[0],
                         lowest_voxel[1]:lowest_voxel[1]  + patch_size[1],
                         lowest_voxel[2]:lowest_voxel[2]  + patch_size[2]]

        if np.count_nonzero(patch) > threshold:
            positive_voxels.append(lowest_voxel)


    return positive_voxels

