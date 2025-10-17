import torch
import torch.nn as nn
import torch.nn.functional as F

class VALoss(nn.Module):
    """Loss for Option A: Valence/Arousal Only"""
    def __init__(self, valence_weight=1.0, arousal_weight=1.0):
        super(VALoss, self).__init__()
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()  # Additional metric
    
    def forward(self, predictions, targets):
        valence_mse = self.mse_loss(predictions['valence'], targets['valence'])
        arousal_mse = self.mse_loss(predictions['arousal'], targets['arousal'])
        
        # Additional MAE for interpretability
        valence_mae = self.mae_loss(predictions['valence'], targets['valence'])
        arousal_mae = self.mae_loss(predictions['arousal'], targets['arousal'])
        
        total_loss = (self.valence_weight * valence_mse + 
                     self.arousal_weight * arousal_mse)
        
        return {
            'total_loss': total_loss,
            'valence_mse': valence_mse,
            'arousal_mse': arousal_mse,
            'valence_mae': valence_mae,
            'arousal_mae': arousal_mae
        }

class EmotionLoss(nn.Module):
    """Loss for Option B: Discrete Emotions Only"""
    def __init__(self, class_weights=None, label_smoothing=0.1):
        super(EmotionLoss, self).__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing
        )
    
    def forward(self, predictions, targets):
        emotion_loss = self.ce_loss(predictions['emotions'], targets['emotions'])
        
        # Calculate accuracy
        pred_classes = predictions['emotions'].argmax(dim=1)
        accuracy = (pred_classes == targets['emotions']).float().mean()
        
        return {
            'total_loss': emotion_loss,
            'emotion_loss': emotion_loss,
            'emotion_accuracy': accuracy
        }

class MultiTaskLoss(nn.Module):
    """Loss for Option C: Multi-Task Learning - The Complete System!"""
    def __init__(self, valence_weight=1.0, arousal_weight=1.0, 
                 emotion_weight=1.0, class_weights=None, 
                 adaptive_weights=True, label_smoothing=0.1):
        super(MultiTaskLoss, self).__init__()
        
        # Loss weights
        self.valence_weight = nn.Parameter(torch.tensor(valence_weight))
        self.arousal_weight = nn.Parameter(torch.tensor(arousal_weight))
        self.emotion_weight = nn.Parameter(torch.tensor(emotion_weight))
        
        # Whether to learn adaptive weights
        self.adaptive_weights = adaptive_weights
        if not adaptive_weights:
            self.valence_weight.requires_grad = False
            self.arousal_weight.requires_grad = False
            self.emotion_weight.requires_grad = False
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    
    def forward(self, predictions, targets):
        # Individual losses
        valence_mse = self.mse_loss(predictions['valence'], targets['valence'])
        arousal_mse = self.mse_loss(predictions['arousal'], targets['arousal'])
        emotion_ce = self.ce_loss(predictions['emotions'], targets['emotions'])
        
        # Additional metrics
        valence_mae = self.mae_loss(predictions['valence'], targets['valence'])
        arousal_mae = self.mae_loss(predictions['arousal'], targets['arousal'])
        
        pred_classes = predictions['emotions'].argmax(dim=1)
        emotion_accuracy = (pred_classes == targets['emotions']).float().mean()
        
        # Weighted combination
        if self.adaptive_weights:
            # Uncertainty-based weighting (learning to balance tasks)
            total_loss = (torch.exp(-self.valence_weight) * valence_mse + self.valence_weight +
                         torch.exp(-self.arousal_weight) * arousal_mse + self.arousal_weight +
                         torch.exp(-self.emotion_weight) * emotion_ce + self.emotion_weight)
        else:
            # Fixed weights
            total_loss = (self.valence_weight * valence_mse + 
                         self.arousal_weight * arousal_mse +
                         self.emotion_weight * emotion_ce)
        
        return {
            'total_loss': total_loss,
            'valence_mse': valence_mse,
            'arousal_mse': arousal_mse,
            'emotion_loss': emotion_ce,
            'valence_mae': valence_mae,
            'arousal_mae': arousal_mae,
            'emotion_accuracy': emotion_accuracy,
            'loss_weights': {
                'valence': self.valence_weight.item(),
                'arousal': self.arousal_weight.item(),
                'emotion': self.emotion_weight.item()
            }
        }

def get_loss_function(model_type='multitask', **kwargs):
    """Factory function for loss functions"""
    if model_type == 'va_only':
        return VALoss(**kwargs)
    elif model_type == 'emotions_only':
        return EmotionLoss(**kwargs)
    elif model_type == 'multitask':
        return MultiTaskLoss(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
