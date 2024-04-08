import time
import logging
import argparse
from pathlib import Path
import yaml
import gc
import random
import gc
from types import SimpleNamespace
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np

from model.model import TimmSegModel, convert_3d
from training_utils import AverageMeter, timeSince
from datasets.augmentations import get_augmentation
from datasets.train_dataset import TrainKidney3DDataset
from datasets.valid_dataset import ValidKidney3DDataset
import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
standard_data_path = Path('./configs/baseline.yaml')
OUTPUT_DIR = Path('./checkpoints')
data_dir = Path('./data/')

patch_size = (32,128,128)
stride = (32,64,64)

def train_fn(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer,
    epoch: int,
    scheduler,
    device,
    best_score,
    CFG,
) -> torch.Tensor:
    """Trains model on train loader. Saves best score."""

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (batch) in enumerate(train_loader):
        
        inputs = batch['image'].to(device)
        labels = batch['mask'].to(device)

        batch_size = labels.size(0)
        
        # Use amp if enabled
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        scheduler.step()
 
        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            logging.info(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
            
            
                
        if CFG.eval_step_save_start_epoch <= epoch and (
            (step + 1) % CFG.eval_freq == 0
        ):
            val_loss = valid_fn(valid_loader, model, criterion, device, CFG)
            score = val_loss
            if score < best_score:
                best_score = score
                save_ckpt(model, CFG)
                logging.info(f"Saving New Best Score Model")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, best_score


@torch.inference_mode()
def valid_fn(
    valid_loader: DataLoader,
    model: nn.Module, 
    criterion: nn.Module, 
    device,
    CFG,
) -> tuple[torch.Tensor, np.ndarray]:
    """Validates model on valid loader."""
    losses = AverageMeter()
    model.eval()
    start = end = time.time()
    for step, (batch) in enumerate(valid_loader):
        inputs = batch['image'].to(device)
        labels = batch['mask'].to(device)
        batch_size = labels.size(0)
        
        y_preds = model(inputs)
            
        loss = criterion(y_preds, labels)
    
            
        losses.update(loss.item(), batch_size)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            logging.info(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )


    model.train()
    return losses.avg

def save_ckpt(
    model: torch.nn.Module,
    CFG
) -> None:
    """Saves checkpoint of model"""
    save_path = OUTPUT_DIR + f'/{CFG.ckpt_name}.pth'

    torch.save(
        {"model": model.state_dict()},
        save_path,
    )

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description='Train model on 3D kidney.')

    parser.add_argument('--config', '-C', 
                        type=str, default=standard_data_path, 
                        help='yaml file path for model config')
    parser.add_argument('--data_dir', '-D', 
                    type=str, default=data_dir, 
                    help='where to load and save data')
    
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Convert dic to namespace
    config_obj = SimpleNamespace(**config_dict)
    
    return config_obj

if __name__ == '__main__':
    args = get_args()

    CFG = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    seed_everything(seed=42)

    # Load data
    kidney3 = np.load(args.data_dir / f'kidney3_dense.npz')
    kidney3_volume = kidney3['volume'].astype(np.uint8)
    kidney3_masks = kidney3['mask'].astype(np.uint8)

    kidney1 = np.load(args.data_dir / f'kidney1_dense.npz')
    kidney1_volume = kidney1['volume'].astype(np.uint8)
    kidney1_masks = kidney1['mask'].astype(np.uint8)

    # Load model
    model = TimmSegModel(CFG.backbone)
    model = convert_3d(model)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {total_params}")

    aug = get_augmentation()

    train_dataset = TrainKidney3DDataset(
        patches=kidney1_volume,
        masks=kidney1_masks,
        patch_size=patch_size,
        transformations=get_augmentation(),
    )

    valid_dataset = ValidKidney3DDataset(
        patches=kidney3_volume,
        masks=kidney3_masks,
        patch_size=patch_size,
        stride_size=(64, 64, 64),
    )


    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.infer_batch_size, shuffle=False)

    # Loss function and optimizer
    # criterion = BCELoss(pos_weight=120,device=device)
    criterion = losses.DiceLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.decoder_lr)
    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train_steps//2)
    best_score = np.inf

    for epoch in range(CFG.epochs):
        start_time = time.time()
        avg_loss, best_score = train_fn(
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        epoch,
        scheduler,
        device,
        best_score,
        CFG
        )
                
                
        avg_val_loss = valid_fn(valid_loader, model, criterion, device, CFG)
        elapsed = time.time() - start_time
        save_ckpt(model, CFG)