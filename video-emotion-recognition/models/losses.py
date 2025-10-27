import torch
import torch.nn as nn
import torch.nn.functional as F

def _align_framewise(pred, tgt):
    """
    If tensors are (B,T), align time length; otherwise return as-is.
    Returns (pred, tgt, is_sequence: bool)
    """
    if pred.dim() == 2 and tgt.dim() == 2:
        T = min(pred.size(1), tgt.size(1))
        return pred[:, :T], tgt[:, :T], True
    return pred, tgt, False


class VALoss(nn.Module):
    """
    Valence/Arousal regression loss.

    Accepts optional/extra kwargs so it won't crash if the caller passes keys
    intended for other losses (e.g., emotion_weight).
    """
    def __init__(self, valence_weight=1.0, arousal_weight=1.0, **_ignore):
        super().__init__()
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, predictions, targets):
        v_pred, v_tgt, v_seq = _align_framewise(predictions['valence'], targets['valence'])
        a_pred, a_tgt, a_seq = _align_framewise(predictions['arousal'], targets['arousal'])

        valence_mse = self.mse(v_pred, v_tgt)
        arousal_mse = self.mse(a_pred, a_tgt)
        total = self.valence_weight * valence_mse + self.arousal_weight * arousal_mse

        # MAE for logging (mean over time if sequence)
        if v_seq:
            v_mae = (v_pred - v_tgt).abs().mean()
            a_mae = (a_pred - a_tgt).abs().mean()
        else:
            v_mae = self.mae(v_pred, v_tgt)
            a_mae = self.mae(a_pred, a_tgt)

        return {
            'total_loss': total,
            'valence_mse': valence_mse,
            'arousal_mse': arousal_mse,
            'valence_mae': v_mae,
            'arousal_mae': a_mae
        }


class EmotionLoss(nn.Module):
    """
    Discrete emotion classification loss.

    Tolerates extra kwargs (e.g., valence_weight) to avoid TypeError when the
    caller forwards a shared kwargs dict.
    """
    def __init__(self, class_weights=None, label_smoothing=0.1, **_ignore):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, predictions, targets):
        ce = self.ce(predictions['emotions'], targets['emotions'])
        acc = (predictions['emotions'].argmax(dim=1) == targets['emotions']).float().mean()
        return {'total_loss': ce, 'emotion_loss': ce, 'emotion_accuracy': acc}


class MultiTaskLoss(nn.Module):
    """
    Multitask loss with optional adaptive (learnable uncertainty) weights.
    """
    def __init__(self, valence_weight=1.0, arousal_weight=1.0, emotion_weight=1.0,
                 class_weights=None, adaptive_weights=True, label_smoothing=0.1, **_ignore):
        super().__init__()
        # learnable uncertainty weights if adaptive
        self.adaptive = adaptive_weights
        if adaptive_weights:
            self.wv = nn.Parameter(torch.tensor(valence_weight))
            self.wa = nn.Parameter(torch.tensor(arousal_weight))
            self.we = nn.Parameter(torch.tensor(emotion_weight))
        else:
            self.register_buffer('wv', torch.tensor(valence_weight))
            self.register_buffer('wa', torch.tensor(arousal_weight))
            self.register_buffer('we', torch.tensor(emotion_weight))

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, predictions, targets):
        v_pred, v_tgt, v_seq = _align_framewise(predictions['valence'], targets['valence'])
        a_pred, a_tgt, a_seq = _align_framewise(predictions['arousal'], targets['arousal'])
        emo_ce = self.ce(predictions['emotions'], targets['emotions'])

        v_mse = self.mse(v_pred, v_tgt)
        a_mse = self.mse(a_pred, a_tgt)

        if self.adaptive:
            total = (torch.exp(-self.wv) * v_mse + self.wv +
                     torch.exp(-self.wa) * a_mse + self.wa +
                     torch.exp(-self.we) * emo_ce + self.we)
        else:
            total = self.wv * v_mse + self.wa * a_mse + self.we * emo_ce

        # MAE logs
        if v_seq:
            v_mae = (v_pred - v_tgt).abs().mean()
            a_mae = (a_pred - a_tgt).abs().mean()
        else:
            v_mae = self.mae(v_pred, v_tgt)
            a_mae = self.mae(a_pred, a_tgt)

        return {
            'total_loss': total,
            'valence_mse': v_mse,
            'arousal_mse': a_mse,
            'emotion_loss': emo_ce,
            'valence_mae': v_mae,
            'arousal_mae': a_mae
        }


def get_loss_function(model_type='multitask', **kwargs):
    """
    Factory that filters kwargs per model type to avoid passing unexpected args.
    Also safe because VALoss/EmotionLoss tolerate extra kwargs.
    """
    if model_type == 'va_only':
        return VALoss(
            valence_weight=kwargs.get('valence_weight', 1.0),
            arousal_weight=kwargs.get('arousal_weight', 1.0),
        )
    elif model_type == 'emotions_only':
        return EmotionLoss(
            class_weights=kwargs.get('class_weights', None),
            label_smoothing=kwargs.get('label_smoothing', 0.1),
        )
    elif model_type == 'multitask':
        return MultiTaskLoss(
            valence_weight=kwargs.get('valence_weight', 1.0),
            arousal_weight=kwargs.get('arousal_weight', 1.0),
            emotion_weight=kwargs.get('emotion_weight', 1.0),
            class_weights=kwargs.get('class_weights', None),
            adaptive_weights=kwargs.get('adaptive_weights', True),
            label_smoothing=kwargs.get('label_smoothing', 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
