import torch
import torch.nn as nn
import torch.nn.functional as F

class RCL_block(nn.Module):
    """
    A Recurrent Convolutional Layer block, as described by the user's Keras code.
    It unrolls the recurrence for a fixed number of timesteps.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        t_steps (int): Number of recurrent timesteps to unroll.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, t_steps=3):
        super(RCL_block, self).__init__()
        self.t_steps = t_steps
        
        # The feed-forward convolution
        self.feed_forward_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        
        # The recurrent convolution layer whose weights are shared across timesteps
        self.recurrent_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        
        self.bn_ff = nn.BatchNorm2d(out_channels)
        self.bn_rc = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Initial feed-forward pass
        x_ff = self.bn_ff(self.feed_forward_conv(x))
        
        # Initialize the recurrent state with the feed-forward output
        x_recurrent = F.relu(x_ff)

        # Unroll the recurrent steps
        for t in range(self.t_steps):
            # Add the original feed-forward state (like a residual connection)
            # This is the key to matching the Keras Add() layer logic
            recurrent_input = x_recurrent + x_ff
            x_recurrent = self.bn_rc(self.recurrent_conv(recurrent_input))
            x_recurrent = F.relu(x_recurrent)
            
        return x_recurrent

class RCNN(nn.Module):
    """
    The full RCNN model architecture.
    """
    def __init__(self, num_classes=10):
        super(RCNN, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, padding='same')
        
        # Recurrent Convolutional Layers (RCL)
        self.rconv1 = RCL_block(192, 192)
        self.dropout1 = nn.Dropout(0.2)
        
        self.rconv2 = RCL_block(192, 192)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        self.rconv3 = RCL_block(192, 192)
        self.dropout3 = nn.Dropout(0.2)

        self.rconv4 = RCL_block(192, 192)
        
        # Classifier part
        # AdaptiveAvgPool2d is a more robust way to handle the final pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = self.rconv1(x)
        x = self.dropout1(x)
        
        x = self.rconv2(x)
        x = self.pool1(x)
        x = self.dropout2(x)
        
        x = self.rconv3(x)
        x = self.dropout3(x)
        
        x = self.rconv4(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x

