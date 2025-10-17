# training/metrics.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionMetrics:
    """Comprehensive metrics for emotion recognition evaluation"""
    
    def __init__(self, model_type='multitask', emotion_names=None):
        self.model_type = model_type
        self.emotion_names = emotion_names or []
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = {
            'valence': [],
            'arousal': [],
            'emotions': []
        }
        self.targets = {
            'valence': [],
            'arousal': [],
            'emotions': []
        }
    
    def update(self, predictions, targets):
        """Update metrics with batch results"""
        # Handle different model types
        if 'valence' in predictions:
            self.predictions['valence'].extend(predictions['valence'].cpu().numpy())
            self.targets['valence'].extend(targets['valence'].cpu().numpy())
        
        if 'arousal' in predictions:
            self.predictions['arousal'].extend(predictions['arousal'].cpu().numpy())
            self.targets['arousal'].extend(targets['arousal'].cpu().numpy())
        
        if 'emotions' in predictions:
            pred_emotions = predictions['emotions'].argmax(dim=1).cpu().numpy()
            self.predictions['emotions'].extend(pred_emotions)
            self.targets['emotions'].extend(targets['emotions'].cpu().numpy())
    
    def compute_va_metrics(self):
        """Compute valence/arousal regression metrics"""
        if not self.predictions['valence']:
            return {}
        
        val_pred = np.array(self.predictions['valence'])
        val_true = np.array(self.targets['valence'])
        aro_pred = np.array(self.predictions['arousal'])
        aro_true = np.array(self.targets['arousal'])
        
        # Mean Squared Error
        val_mse = np.mean((val_pred - val_true) ** 2)
        aro_mse = np.mean((aro_pred - aro_true) ** 2)
        
        # Mean Absolute Error
        val_mae = np.mean(np.abs(val_pred - val_true))
        aro_mae = np.mean(np.abs(aro_pred - aro_true))
        
        # Pearson Correlation
        val_corr, _ = pearsonr(val_pred, val_true)
        aro_corr, _ = pearsonr(aro_pred, aro_true)
        
        # R-squared
        val_r2 = 1 - (np.sum((val_true - val_pred) ** 2) / 
                      np.sum((val_true - np.mean(val_true)) ** 2))
        aro_r2 = 1 - (np.sum((aro_true - aro_pred) ** 2) / 
                      np.sum((aro_true - np.mean(aro_true)) ** 2))
        
        return {
            'valence_mse': val_mse,
            'valence_mae': val_mae,
            'valence_corr': val_corr,
            'valence_r2': val_r2,
            'arousal_mse': aro_mse,
            'arousal_mae': aro_mae,
            'arousal_corr': aro_corr,
            'arousal_r2': aro_r2,
            'combined_mse': (val_mse + aro_mse) / 2,
            'combined_mae': (val_mae + aro_mae) / 2
        }
    
    def compute_emotion_metrics(self):
        """Compute discrete emotion classification metrics"""
        if not self.predictions['emotions']:
            return {}
        
        pred = np.array(self.predictions['emotions'])
        true = np.array(self.targets['emotions'])
        
        # Basic metrics
        accuracy = accuracy_score(true, pred)
        f1_macro = f1_score(true, pred, average='macro', zero_division=0)
        f1_weighted = f1_score(true, pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        f1_per_class = f1_score(true, pred, average=None, zero_division=0)
        
        return {
            'emotion_accuracy': accuracy,
            'emotion_f1_macro': f1_macro,
            'emotion_f1_weighted': f1_weighted,
            'emotion_f1_per_class': f1_per_class.tolist()
        }
    
    def compute_all_metrics(self):
        """Compute all relevant metrics based on model type"""
        metrics = {}
        
        if self.model_type in ['va_only', 'multitask']:
            metrics.update(self.compute_va_metrics())
        
        if self.model_type in ['emotions_only', 'multitask']:
            metrics.update(self.compute_emotion_metrics())
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix for emotion classification"""
        if not self.predictions['emotions']:
            return None
        
        pred = np.array(self.predictions['emotions'])
        true = np.array(self.targets['emotions'])
        
        cm = confusion_matrix(true, pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.emotion_names[:cm.shape[1]], 
                   yticklabels=self.emotion_names[:cm.shape[0]])
        plt.title('Emotion Classification Confusion Matrix')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return plt.gcf()
    
    def plot_va_scatter(self, save_path=None):
        """Plot valence/arousal prediction scatter plots"""
        if not self.predictions['valence']:
            return None
        
        val_pred = np.array(self.predictions['valence'])
        val_true = np.array(self.targets['valence'])
        aro_pred = np.array(self.predictions['arousal'])
        aro_true = np.array(self.targets['arousal'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Valence scatter plot
        ax1.scatter(val_true, val_pred, alpha=0.6)
        ax1.plot([val_true.min(), val_true.max()], [val_true.min(), val_true.max()], 'r--')
        ax1.set_xlabel('True Valence')
        ax1.set_ylabel('Predicted Valence')
        ax1.set_title(f'Valence Prediction (r={pearsonr(val_pred, val_true)[0]:.3f})')
        
        # Arousal scatter plot
        ax2.scatter(aro_true, aro_pred, alpha=0.6)
        ax2.plot([aro_true.min(), aro_true.max()], [aro_true.min(), aro_true.max()], 'r--')
        ax2.set_xlabel('True Arousal')
        ax2.set_ylabel('Predicted Arousal')
        ax2.set_title(f'Arousal Prediction (r={pearsonr(aro_pred, aro_true)[0]:.3f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
