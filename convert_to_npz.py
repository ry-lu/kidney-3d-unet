import argparse
import os
import logging
from pathlib import Path
import cv2
import numpy as np

data_dir = Path('./data/')
patch_size = (128, 256, 256)
stride_size = (128, 128, 128)

def get_data(folder):
    kidney_paths =  sorted(folder.glob('*.tif'))
    kidney_paths = [str(posix_path) for posix_path in kidney_paths]
    return kidney_paths

def read_volume(path, is_mask=False):
    """Reads grayscale images into np.uint8 and stacks them."""
    volume = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.uint8) for f in path]
    volume = np.stack(volume).astype(np.uint8)
    if is_mask:
        # Converts to binary mask
        volume = np.where(volume > 0.5, 1, 0).astype(np.uint8)
    return volume

def save_kidney_patches(image_path, mask_path, kidney_name):
    """Saves kidney data into npz format."""
    if len(image_path) == 0 or len(mask_path) == 0:
        raise ValueError("No images or masks found.")
    
    if len(image_path) != len(mask_path):
        raise ValueError("Number of images must equal number of masks.")

    volume = read_volume(image_path)
    mask = read_volume(mask_path,is_mask=True)

    os.makedirs(data_dir / f'{kidney_name}_patches', exist_ok=True)
    file_path = os.path.join(data_dir / f'{kidney_name}_dense.npz')
    np.savez(file_path, volume=volume,mask=mask)

def get_args():
    parser = argparse.ArgumentParser(description='Convert data to npz format.')
    parser.add_argument('--data_dir', '-D', type=str, default=data_dir, help='where to load and save data')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    # Find kidney3 data (depends on data format)
    folder = args.data_dir / "train" / "kidney_3_sparse" / "images"
    kidney3_dense_paths = get_data(folder)

    folder = args.data_dir / "train" / "kidney_3_dense" / "labels"
    kidney3_dense_mask_paths = get_data(folder)


    # Filter out kidney3 sparse labels (competition data format)
    kidney3_dense_paths = [
        img 
        for img in kidney3_dense_paths 
        if any(
            os.path.splitext(os.path.basename(mask))[0] in img 
            for mask in kidney3_dense_mask_paths
        )
    ]

    save_kidney_patches(kidney3_dense_paths,kidney3_dense_mask_paths,"kidney3")
        
    # Find kidney1 data 
    folder = args.data_dir / "train" / "kidney_1_dense" / "images"
    kidney1_dense_paths = get_data(folder)

    folder = args.data_dir / "train" / "kidney_1_dense" / "labels"
    kidney1_dense_mask_paths = get_data(folder)

    save_kidney_patches(kidney1_dense_paths,kidney1_dense_mask_paths,"kidney1")
