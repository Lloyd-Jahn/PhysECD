"""
模型训练脚本

运行指令：
cd 到 PhysECD 项目根目录下，执行：
python scripts/03_train.py
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# 添加父目录到系统路径，以确保能正确导入 physecd 模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physecd.models import PhysECDModel
from physecd.physics import PhysECDLoss


def setup_logging(log_dir):
    """初始化日志配置，同时输出到文件和控制台。"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_data(data_dir, batch_size=64, num_workers=16):
    """加载训练集和验证集。"""
    logger = logging.getLogger(__name__)

    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")

    logger.info(f"Loading training data from {train_path}")
    train_data = torch.load(train_path, weights_only=False)
    logger.info(f"Loaded {len(train_data)} training samples")

    logger.info(f"Loading validation data from {val_path}")
    val_data = torch.load(val_path, weights_only=False)
    logger.info(f"Loaded {len(val_data)} validation samples")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def compute_total_loss(loss_dict, config):
    """根据配置中的权重计算总损失。"""
    return (
        config["weight_shape"] * loss_dict["loss_shape"]
        + config["weight_mag"] * loss_dict["loss_mag"]
        + config["weight_E"] * loss_dict["loss_E"]
        + config["weight_reg"] * loss_dict["loss_reg"]
    )


def train_epoch(model, loader, criterion, optimizer, device, config, logger):
    """训练一个 epoch，返回各项损失平均值。"""
    model.train()

    metrics = {
        "loss": 0.0,
        "loss_shape": 0.0,
        "loss_mag": 0.0,
        "loss_E": 0.0,
        "loss_reg": 0.0,
    }
    num_batches = 0
    start_time = time.time()

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        pred = model(data)
        loss_dict = criterion(pred, data)
        loss = compute_total_loss(loss_dict, config)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics["loss"] += loss.item()
        for key in ("loss_shape", "loss_mag", "loss_E", "loss_reg"):
            metrics[key] += loss_dict[key].item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Batch [{batch_idx + 1}/{len(loader)}] "
                f"Total Loss: {metrics['loss'] / num_batches:.4f} | "
                f"Shape: {loss_dict['loss_shape'].item():.4f}, "
                f"Mag: {loss_dict['loss_mag'].item():.4f}, "
                f"E: {loss_dict['loss_E'].item():.4f}"
            )

    for key in metrics:
        metrics[key] /= num_batches

    elapsed = time.time() - start_time
    logger.info(f"  Training epoch completed in {elapsed:.2f}s")
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, config, logger):
    """在验证集上评估模型，返回各项损失平均值。"""
    model.eval()

    metrics = {
        "loss": 0.0,
        "loss_shape": 0.0,
        "loss_mag": 0.0,
        "loss_E": 0.0,
        "loss_reg": 0.0,
    }
    num_batches = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)

        loss_dict = criterion(pred, data)
        loss = compute_total_loss(loss_dict, config)

        metrics["loss"] += loss.item()
        for key in ("loss_shape", "loss_mag", "loss_E", "loss_reg"):
            metrics[key] += loss_dict[key].item()
        num_batches += 1

    for key in metrics:
        metrics[key] /= num_batches
    return metrics


def plot_loss_curves(train_losses, val_losses, save_path):
    """绘制损失曲线（2x2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses["loss"]) + 1)

    axes[0, 0].plot(epochs, train_losses["loss"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_losses["loss"], "r-", label="Validation", linewidth=2)
    axes[0, 0].set_title("Total Weighted Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(epochs, train_losses["loss_shape"], "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, val_losses["loss_shape"], "r-", label="Validation", linewidth=2)
    axes[0, 1].set_title("Shape Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Distance (0 to 2)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, max(2.0, max(val_losses["loss_shape"])))

    axes[1, 0].plot(epochs, train_losses["loss_mag"], "b-", label="Train", linewidth=2)
    axes[1, 0].plot(epochs, val_losses["loss_mag"], "r-", label="Validation", linewidth=2)
    axes[1, 0].set_title("Magnitude Loss (Smooth L1)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    axes[1, 1].plot(epochs, train_losses["loss_E"], "b-", label="Train", linewidth=2)
    axes[1, 1].plot(epochs, val_losses["loss_E"], "r-", label="Validation", linewidth=2)
    axes[1, 1].set_title("Excitation Energy Loss (MSE)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """主训练函数。"""
    config = {
        "data_dir": "data/processed_with_enantiomers",
        "checkpoint_dir": "checkpoints",
        "batch_size": 64,
        "num_epochs": 2000,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "num_workers": 16,
        "num_features": 128,
        "max_l": 3,
        "num_blocks": 3,
        "num_radial": 32,
        "cutoff": 50.0,
        "n_states": 20,
        "max_atomic_number": 60,
        "sigma": 0.4,
        "weight_shape": 1.0,
        "weight_mag": 1.0,
        "weight_E": 0.1,
        "weight_reg": 1e-4,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(config["checkpoint_dir"])

    logger.info("=" * 80)
    logger.info("PhysECD Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config}")

    logger.info("\nLoading data...")
    train_loader, val_loader = load_data(
        config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    logger.info("\nInitializing model...")
    model = PhysECDModel(
        num_features=config["num_features"],
        max_l=config["max_l"],
        num_blocks=config["num_blocks"],
        num_radial=config["num_radial"],
        cutoff=config["cutoff"],
        n_states=config["n_states"],
        max_atomic_number=config["max_atomic_number"],
    ).to(device)

    num_params = model.get_num_params()
    logger.info(f"Model initialized with {num_params:,} trainable parameters")

    criterion = PhysECDLoss(sigma=config["sigma"]).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        eta_min=1e-5,
        T_max=config["num_epochs"] // 5,
    )

    best_val_shape_loss = float("inf")
    metric_keys = ["loss", "loss_shape", "loss_mag", "loss_E", "loss_reg"]
    train_losses = {key: [] for key in metric_keys}
    val_losses = {key: [] for key in metric_keys}

    logger.info("\nStarting training...")
    logger.info("=" * 80)

    for epoch in range(1, config["num_epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}")
        logger.info("-" * 80)

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config, logger
        )

        logger.info("  Running validation...")
        val_metrics = validate(
            model, val_loader, criterion, device, config, logger
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info("-" * 80)
        logger.info(
            f"Epoch {epoch} Summary - LR: {current_lr:.6f}\n"
            f"  Train -> Total: {train_metrics['loss']:.4f} | "
            f"Shape: {train_metrics['loss_shape']:.4f} | "
            f"Mag: {train_metrics['loss_mag']:.4f} | "
            f"E: {train_metrics['loss_E']:.4f}\n"
            f"  Val   -> Total: {val_metrics['loss']:.4f} | "
            f"Shape: {val_metrics['loss_shape']:.4f} | "
            f"Mag: {val_metrics['loss_mag']:.4f} | "
            f"E: {val_metrics['loss_E']:.4f}"
        )

        for key in metric_keys:
            train_losses[key].append(train_metrics[key])
            val_losses[key].append(val_metrics[key])

        if val_metrics["loss_shape"] < best_val_shape_loss:
            best_val_shape_loss = val_metrics["loss_shape"]
            checkpoint_path = os.path.join(config["checkpoint_dir"], "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_shape_loss": val_metrics["loss_shape"],
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(
                f"  >>> Saved new best model (val Shape Loss: {best_val_shape_loss:.4f})"
            )

        if epoch % 100 == 0:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"  Saved checkpoint at epoch {epoch}")

        if epoch % 5 == 0 or epoch == 1:
            plot_path = os.path.join(config["checkpoint_dir"], "loss_curves.png")
            plot_loss_curves(train_losses, val_losses, plot_path)
            logger.info("  Updated loss curves plot")

    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Best Val Shape Loss: {best_val_shape_loss:.4f} (Closer to 0 is better)")
    logger.info(
        f"Best model saved to: {os.path.join(config['checkpoint_dir'], 'best_model.pt')}"
    )
    logger.info("=" * 80)

    plot_path = os.path.join(config["checkpoint_dir"], "loss_curves_final.png")
    plot_loss_curves(train_losses, val_losses, plot_path)


if __name__ == "__main__":
    main()
