"""
Training script for PhysECD model.

Implements complete training pipeline with:
- Data loading
- Model initialization
- Training loop with validation
- Checkpointing
- Loss visualization
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.models import PhysECDModel
from physecd.physics import PhysECDLoss


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(data_dir, batch_size=32, num_workers=4):
    """Load training and validation datasets."""
    logger = logging.getLogger(__name__)

    train_path = os.path.join(data_dir, 'train.pt')
    val_path = os.path.join(data_dir, 'val.pt')

    logger.info(f"Loading training data from {train_path}")
    train_data = torch.load(train_path, weights_only=False)
    logger.info(f"Loaded {len(train_data)} training samples")

    logger.info(f"Loading validation data from {val_path}")
    val_data = torch.load(val_path, weights_only=False)
    logger.info(f"Loaded {len(val_data)} validation samples")

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, logger):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    loss_components = {'loss_E': 0.0, 'loss_mu': 0.0, 'loss_m': 0.0, 'loss_R': 0.0}
    num_batches = 0

    start_time = time.time()

    for batch_idx, data in enumerate(loader):
        # Move data to device
        data = data.to(device)

        # Forward pass
        pred = model(data)

        # Prepare target dictionary
        target = {
            'y_E': data.y_E,
            'y_mu_vel': data.y_mu_vel,
            'y_m': data.y_m,
            'y_R': data.y_R
        }

        # Compute loss
        loss, loss_dict = criterion(pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += loss_dict['loss']
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        num_batches += 1

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"  Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {avg_loss:.4f} "
                f"(E: {loss_dict['loss_E']:.4f}, "
                f"mu: {loss_dict['loss_mu']:.4f}, "
                f"m: {loss_dict['loss_m']:.4f}, "
                f"R: {loss_dict['loss_R']:.4f})"
            )

    # Compute epoch averages
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    elapsed = time.time() - start_time
    logger.info(f"  Training epoch completed in {elapsed:.2f}s")

    return avg_loss, loss_components


@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    loss_components = {'loss_E': 0.0, 'loss_mu': 0.0, 'loss_m': 0.0, 'loss_R': 0.0}
    num_batches = 0

    for data in loader:
        # Move data to device
        data = data.to(device)

        # Forward pass
        pred = model(data)

        # Prepare target dictionary
        target = {
            'y_E': data.y_E,
            'y_mu_vel': data.y_mu_vel,
            'y_m': data.y_m,
            'y_R': data.y_R
        }

        # Compute loss
        loss, loss_dict = criterion(pred, target)

        # Accumulate losses
        total_loss += loss_dict['loss']
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        num_batches += 1

    # Compute averages
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(15, 10))

    # Main plot: total loss
    plt.subplot(2, 3, 1)
    epochs = range(1, len(train_losses['total']) + 1)
    plt.plot(epochs, train_losses['total'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['total'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot: Energy loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_losses['E'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['E'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Excitation Energy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot: Electric dipole loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_losses['mu'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['mu'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Electric Dipole Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot: Magnetic dipole loss
    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_losses['m'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['m'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Magnetic Dipole Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot: Rotatory strength loss
    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_losses['R'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, val_losses['R'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Rotatory Strength Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_dir': 'data/processed_with_enantiomers',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 32,
        'num_epochs': 25,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_workers': 4,
        # Model hyperparameters
        'num_features': 128,
        'max_l': 2,
        'num_blocks': 3,
        'num_radial': 32,
        'cutoff': 5.0,
        'n_states': 20,
        'max_atomic_number': 60,
        
        # ==========================================
        # Loss Weights（归一化后的相对权重）
        # ==========================================
        'lambda_E': 1.0,
        'lambda_mu': 1.0,
        'lambda_m': 1.0,
        'lambda_R': 1.0,
    }

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config['checkpoint_dir'])

    logger.info("=" * 80)
    logger.info("PhysECD Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config}")

    # Load data
    logger.info("\nLoading data...")
    train_loader, val_loader = load_data(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Initialize model
    logger.info("\nInitializing model...")
    model = PhysECDModel(
        num_features=config['num_features'],
        max_l=config['max_l'],
        num_blocks=config['num_blocks'],
        num_radial=config['num_radial'],
        cutoff=config['cutoff'],
        n_states=config['n_states'],
        max_atomic_number=config['max_atomic_number']
    ).to(device)

    num_params = model.get_num_params()
    logger.info(f"Model initialized with {num_params:,} trainable parameters")

    # Initialize loss function
    criterion = PhysECDLoss(
        lambda_E=config['lambda_E'],
        lambda_mu=config['lambda_mu'],
        lambda_m=config['lambda_m'],
        lambda_R=config['lambda_R']
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )

    # Training tracking
    best_val_loss = float('inf')
    train_losses = {'total': [], 'E': [], 'mu': [], 'm': [], 'R': []}
    val_losses = {'total': [], 'E': [], 'mu': [], 'm': [], 'R': []}

    # Training loop
    logger.info("\nStarting training...")
    logger.info("=" * 80)

    for epoch in range(1, config['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}")
        logger.info("-" * 80)

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )

        # Validate
        logger.info("  Running validation...")
        val_loss, val_components = validate(
            model, val_loader, criterion, device, logger
        )

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch summary
        logger.info("-" * 80)
        logger.info(
            f"Epoch {epoch} Summary - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        logger.info(
            f"  Train: E={train_components['loss_E']:.4f}, "
            f"mu={train_components['loss_mu']:.4f}, "
            f"m={train_components['loss_m']:.4f}, "
            f"R={train_components['loss_R']:.4f}"
        )
        logger.info(
            f"  Val:   E={val_components['loss_E']:.4f}, "
            f"mu={val_components['loss_mu']:.4f}, "
            f"m={val_components['loss_m']:.4f}, "
            f"R={val_components['loss_R']:.4f}"
        )

        # Store losses for plotting
        train_losses['total'].append(train_loss)
        train_losses['E'].append(train_components['loss_E'])
        train_losses['mu'].append(train_components['loss_mu'])
        train_losses['m'].append(train_components['loss_m'])
        train_losses['R'].append(train_components['loss_R'])

        val_losses['total'].append(val_loss)
        val_losses['E'].append(val_components['loss_E'])
        val_losses['mu'].append(val_components['loss_mu'])
        val_losses['m'].append(val_components['loss_m'])
        val_losses['R'].append(val_components['loss_R'])

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"  Saved checkpoint at epoch {epoch}")

        # Plot loss curves
        if epoch % 5 == 0 or epoch == 1:
            plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves.png')
            plot_loss_curves(train_losses, val_losses, plot_path)
            logger.info(f"  Updated loss curves plot")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {os.path.join(config['checkpoint_dir'], 'best_model.pt')}")
    logger.info("=" * 80)

    # Final plot
    plot_path = os.path.join(config['checkpoint_dir'], 'loss_curves_final.png')
    plot_loss_curves(train_losses, val_losses, plot_path)


if __name__ == '__main__':
    main()
