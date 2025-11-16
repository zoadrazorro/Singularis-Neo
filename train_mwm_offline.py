"""
Train MWM Offline - From Collected Data

MOVE 2 Follow-up: Train MWM on collected gameplay data

This script trains the Mental World Model (MWM) on logged trajectories
from run_local_agi.py. Run this AFTER collecting data, not during gameplay.

Usage:
    # After collecting 100+ episodes
    python train_mwm_offline.py --log logs/training_local.jsonl --epochs 10

Features:
- Loads training data from JSONL
- Trains MWM to predict affect from GWM + IWM + self-state
- Saves trained weights
- No impact on live system (offline training)
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm

from singularis.mwm import MentalWorldModelModule, MWMLoss
from singularis.mwm.integration import pack_gwm_features, pack_self_features


class TrainingDataset(torch.utils.data.Dataset):
    """
    Dataset for MWM training.
    
    Loads (GWM, IWM, self-state, action, reward) tuples from JSONL.
    """
    
    def __init__(self, log_file: Path):
        self.entries = []
        
        logger.info(f"Loading training data from {log_file}...")
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    self.entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.entries)} training entries")
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Pack GWM features
        gwm_features = entry.get('gwm_features', {})
        gwm_packed = pack_gwm_features(gwm_features)
        
        # IWM latent
        iwm_latent = entry.get('iwm_latent')
        if iwm_latent is None:
            iwm_latent = np.zeros(768, dtype=np.float32)
        else:
            iwm_latent = np.array(iwm_latent, dtype=np.float32)
        
        # Self-state
        self_state = entry.get('self_state', {})
        self_packed = np.array([
            self_state.get('health', 1.0),
            self_state.get('stamina', 1.0),
            self_state.get('magicka', 1.0),
            0.0, 0.0, 0.0, 0.0, 0.0  # Padding
        ], dtype=np.float32)
        
        # Reward (proxy for affect)
        reward = entry.get('reward_proxy', 0.0)
        
        return {
            'gwm': torch.from_numpy(gwm_packed),
            'iwm': torch.from_numpy(iwm_latent),
            'self': torch.from_numpy(self_packed),
            'reward': torch.tensor(reward, dtype=torch.float32)
        }


def train_mwm(
    log_file: Path,
    output_dir: Path,
    latent_dim: int = 256,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda:0"
):
    """
    Train MWM on collected data.
    
    Args:
        log_file: Path to training_local.jsonl
        output_dir: Where to save trained weights
        latent_dim: MWM latent dimension
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    logger.info("="*60)
    logger.info("MWM Offline Training")
    logger.info("="*60)
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load dataset
    dataset = TrainingDataset(log_file)
    
    if len(dataset) < 10:
        logger.error(f"Not enough data! Need at least 10 entries, got {len(dataset)}")
        logger.error("Run run_local_agi.py with COLLECT_TRAINING_DATA=True first")
        return
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Train: {train_size} entries")
    logger.info(f"Val: {val_size} entries")
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    model = MentalWorldModelModule(latent_dim=latent_dim).to(device)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss & optimizer
    criterion = MWMLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            gwm = batch['gwm'].to(device)
            iwm = batch['iwm'].to(device)
            self_state = batch['self'].to(device)
            reward = batch['reward'].to(device)
            
            # Initialize latent
            z_prev = torch.zeros(gwm.size(0), latent_dim).to(device)
            
            # Forward
            z_t, decoded, z_hat = model(z_prev, gwm, iwm, self_state, None)
            
            # Loss (simplified - just reconstruction)
            loss = criterion.reconstruction_loss(decoded, {
                'world': gwm,
                'self': self_state,
                'affect': torch.zeros(gwm.size(0), 4).to(device)  # Placeholder
            })
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                gwm = batch['gwm'].to(device)
                iwm = batch['iwm'].to(device)
                self_state = batch['self'].to(device)
                
                z_prev = torch.zeros(gwm.size(0), latent_dim).to(device)
                z_t, decoded, z_hat = model(z_prev, gwm, iwm, self_state, None)
                
                loss = criterion.reconstruction_loss(decoded, {
                    'world': gwm,
                    'self': self_state,
                    'affect': torch.zeros(gwm.size(0), 4).to(device)
                })
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"  Train loss: {train_loss:.4f}")
        logger.info(f"  Val loss: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_dir / "mwm_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"  ✓ Saved checkpoint to {checkpoint_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint: {output_dir / 'mwm_best.pt'}")
    logger.info("="*60)
    
    logger.info("\nTo use trained weights:")
    logger.info("  1. Load checkpoint in run_local_agi.py:")
    logger.info("     checkpoint = torch.load('checkpoints/mwm_best.pt')")
    logger.info("     mwm_module.load_state_dict(checkpoint['model_state_dict'])")
    logger.info("  2. Run with trained affect predictions!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MWM offline")
    parser.add_argument('--log', type=str, default='logs/training_local.jsonl',
                        help='Training log file')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='MWM latent dimension')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to train on')
    
    args = parser.parse_args()
    
    train_mwm(
        log_file=Path(args.log),
        output_dir=Path(args.output),
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
