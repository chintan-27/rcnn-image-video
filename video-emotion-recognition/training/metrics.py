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

        def _flat_or_time_mean(x):
            """
            Accept (B,) or (B,T) tensors; if (B,T), average over T for logging metrics.
            Returns a 1-D numpy array of length B.
            """
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x)
            if x.ndim == 2:  # (B,T)
                return x.mean(axis=1)
            return x  # (B,)

        if 'valence' in predictions and 'valence' in targets:
            self.predictions['valence'].extend(_flat_or_time_mean(predictions['valence']))
            self.targets['valence'].extend(_flat_or_time_mean(targets['valence']))

        if 'arousal' in predictions and 'arousal' in targets:
            self.predictions['arousal'].extend(_flat_or_time_mean(predictions['arousal']))
            self.targets['arousal'].extend(_flat_or_time_mean(targets['arousal']))

        if 'emotions' in predictions and 'emotions' in targets:
            pred_emotions = predictions['emotions'].argmax(dim=1).detach().cpu().numpy()
            self.predictions['emotions'].extend(pred_emotions)
            self.targets['emotions'].extend(
                targets['emotions'].detach().cpu().numpy()
                if torch.is_tensor(targets['emotions']) else np.asarray(targets['emotions'])
            )

    def compute_va_metrics(self):
        """Compute valence/arousal regression metrics"""
        if not self.predictions['valence']:
            return {}

        val_pred = np.asarray(self.predictions['valence']).astype(np.float32)
        val_true = np.asarray(self.targets['valence']).astype(np.float32)
        aro_pred = np.asarray(self.predictions['arousal']).astype(np.float32)
        aro_true = np.asarray(self.targets['arousal']).astype(np.float32)

        # Mean Squared Error
        val_mse = np.mean((val_pred - val_true) ** 2.0)
        aro_mse = np.mean((aro_pred - aro_true) ** 2.0)

        # Mean Absolute Error
        val_mae = np.mean(np.abs(val_pred - val_true))
        aro_mae = np.mean(np.abs(aro_pred - aro_true))

        # Pearson Correlation (guard for constant arrays)
        def _safe_pearson(a, b):
            if np.all(a == a[0]) or np.all(b == b[0]):
                return 0.0
            return float(pearsonr(a, b)[0])

        val_corr = _safe_pearson(val_pred, val_true)
        aro_corr = _safe_pearson(aro_pred, aro_true)

        # R-squared
        def _r2(y_true, y_pred):
            denom = np.sum((y_true - np.mean(y_true)) ** 2.0)
            if denom == 0:
                return 0.0
            return float(1.0 - (np.sum((y_true - y_pred) ** 2.0) / denom))

        val_r2 = _r2(val_true, val_pred)
        aro_r2 = _r2(aro_true, aro_pred)

        return {
            'valence_mse': float(val_mse),
            'valence_mae': float(val_mae),
            'valence_corr': float(val_corr),
            'valence_r2': float(val_r2),
            'arousal_mse': float(aro_mse),
            'arousal_mae': float(aro_mae),
            'arousal_corr': float(aro_corr),
            'arousal_r2': float(aro_r2),
            'combined_mse': float((val_mse + aro_mse) / 2.0),
            'combined_mae': float((val_mae + aro_mae) / 2.0),
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
            'emotion_accuracy': float(accuracy),
            'emotion_f1_macro': float(f1_macro),
            'emotion_f1_weighted': float(f1_weighted),
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

