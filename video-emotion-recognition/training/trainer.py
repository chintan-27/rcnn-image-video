import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import logging
from .metrics import EmotionMetrics

class EmotionTrainer:
    """
    Unified trainer for all three model approaches
    Handles training, validation, checkpointing, and logging
    """

    def __init__(self, model, criterion, optimizer, scheduler=None,
                 device='cuda', model_type='multitask',
                 save_dir='./checkpoints', log_dir='./logs',
                 emotion_names=None):

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_type = model_type

        # Setup directories
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        self.writer = SummaryWriter(log_dir)
        self.logger = self._setup_logger()

        # Metrics tracking
        self.emotion_names = emotion_names
        self.best_metrics = {}
        self.epoch = 0

        # Training state
        self.train_losses = []
        self.val_losses = []

    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('EmotionTrainer')
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if re-instantiated
        if not logger.handlers:
            # File handler
            fh = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
            fh.setLevel(logging.INFO)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()

        epoch_losses = []
        metrics = EmotionMetrics(self.model_type, self.emotion_names)

        start_time = time.time()

        for batch_idx, (videos, targets) in enumerate(train_loader):
            # Move data to device
            videos = videos.to(self.device)
            targets = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in targets.items()}

            # Reshape for 3D CNN: (B, T, C, H, W) -> (B, C, T, H, W)
            videos = videos.permute(0, 2, 1, 3, 4)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(videos)  # VA can be (B,T); keep as-is for loss

            # Compute loss (sequence-aware criterion)
            loss_dict = self.criterion(predictions, targets)
            total_loss = loss_dict['total_loss']

            # Backward pass
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics (metrics internally average time for logging)
            epoch_losses.append(total_loss.item())
            metrics.update(predictions, targets)

            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {total_loss.item():.4f}'
                )

                # Log to tensorboard
                global_step = self.epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', total_loss.item(), global_step)

                # Log individual losses for multitask
                for loss_name, loss_value in loss_dict.items():
                    if loss_name != 'total_loss' and torch.is_tensor(loss_value):
                        self.writer.add_scalar(f'train/{loss_name}', loss_value.item(), global_step)

        # Compute epoch metrics
        epoch_metrics = metrics.compute_all_metrics()
        avg_loss = np.mean(epoch_losses)

        epoch_time = time.time() - start_time

        self.logger.info(f'Epoch {self.epoch} Training - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s')

        return avg_loss, epoch_metrics

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()

        epoch_losses = []
        metrics = EmotionMetrics(self.model_type, self.emotion_names)

        with torch.no_grad():
            for videos, targets in val_loader:
                # Move data to device
                videos = videos.to(self.device)
                targets = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in targets.items()}

                # Reshape for 3D CNN
                videos = videos.permute(0, 2, 1, 3, 4)

                # Forward pass
                predictions = self.model(videos)

                # Compute loss
                loss_dict = self.criterion(predictions, targets)
                total_loss = loss_dict['total_loss']

                # Track metrics
                epoch_losses.append(total_loss.item())
                metrics.update(predictions, targets)

        # Compute epoch metrics
        epoch_metrics = metrics.compute_all_metrics()
        avg_loss = np.mean(epoch_losses)

        self.logger.info(f'Epoch {self.epoch} Validation - Loss: {avg_loss:.4f}')

        # Log detailed metrics
        for metric_name, metric_value in epoch_metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.info(f'  {metric_name}: {metric_value:.4f}')

        return avg_loss, epoch_metrics, metrics

    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_type': self.model_type
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved at epoch {self.epoch}')

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f'Loaded checkpoint from epoch {self.epoch}')
        return checkpoint.get('metrics', {})

    def train(self, train_loader, val_loader, num_epochs,
              early_stopping_patience=10, save_every=5):
        """
        Main training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
            save_every: Save checkpoint every N epochs
        """

        self.logger.info(f'Starting training for {num_epochs} epochs')
        self.logger.info(f'Model type: {self.model_type}')
        self.logger.info(f'Device: {self.device}')

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss, val_metrics, val_metrics_obj = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)

            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.writer.add_scalar(f'epoch/val_{metric_name}', metric_value, epoch)

            # Check for best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.best_metrics = val_metrics.copy()
                patience_counter = 0

                # Save plots for best model
                if self.model_type in ['emotions_only', 'multitask']:
                    fig = val_metrics_obj.plot_confusion_matrix()
                    if fig:
                        self.writer.add_figure('confusion_matrix', fig, epoch)
                        plt.close(fig)

                if self.model_type in ['va_only', 'multitask']:
                    fig = val_metrics_obj.plot_va_scatter()
                    if fig:
                        self.writer.add_figure('va_scatter', fig, epoch)
                        plt.close(fig)
            else:
                patience_counter += 1

            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

        self.logger.info('Training completed!')
        self.logger.info(f'Best validation loss: {best_val_loss:.4f}')

        # Log final best metrics
        for metric_name, metric_value in self.best_metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.info(f'Best {metric_name}: {metric_value:.4f}')

        self.writer.close()

        return self.best_metrics

