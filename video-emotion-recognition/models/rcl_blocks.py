import torch
import torch.nn as nn
import torch.nn.functional as F

class RCL_block_3D(nn.Module):
    """
    3D Recurrent Convolutional Layer - The heart of our model!
    
    This block processes spatiotemporal features recurrently,
    refining them through multiple iterations.
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(3, 3, 3), t_steps=3, 
                 temporal_stride=1, dropout_rate=0.1):
        super(RCL_block_3D, self).__init__()
        
        self.t_steps = t_steps
        self.out_channels = out_channels
        
        # Calculate padding to maintain spatial dimensions
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2, (kernel_size[2]-1)//2)
        
        # Feed-forward 3D convolution (input -> initial features)
        self.feed_forward_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=(temporal_stride, 1, 1),
            padding=padding,
            bias=False  # We'll use batch norm
        )
        
        # Recurrent 3D convolution (features -> refined features)
        self.recurrent_conv = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),  # No striding in recurrent path
            padding=padding,
            bias=False
        )
        
        # Batch normalization layers
        self.bn_ff = nn.BatchNorm3d(out_channels)
        self.bn_rc = nn.BatchNorm3d(out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        nn.init.kaiming_normal_(self.feed_forward_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.recurrent_conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        Forward pass through RCL block
        
        Args:
            x: Input tensor (batch_size, channels, time, height, width)
            
        Returns:
            Refined features after recurrent processing
        """
        # Step 1: Feed-forward pass to get initial features
        x_ff = self.bn_ff(self.feed_forward_conv(x))
        
        # Step 2: Initialize recurrent state with activated feed-forward features
        x_recurrent = F.relu(x_ff)
        
        # Step 3: Recurrent refinement loop - the magic happens here!
        for t in range(self.t_steps):
            # Residual connection: combine current state with original features
            recurrent_input = x_recurrent + x_ff
            
            # Process through recurrent convolution
            x_recurrent = self.recurrent_conv(recurrent_input)
            x_recurrent = self.bn_rc(x_recurrent)
            x_recurrent = F.relu(x_recurrent)
            
            # Apply dropout (only during training)
            if self.training:
                x_recurrent = self.dropout(x_recurrent)
        
        return x_recurrent

class DepthwiseSeparable3D(nn.Module):
    """
    Enhanced Depthwise Separable 3D Convolution for efficiency
    Splits 3D convolution into depthwise + pointwise operations
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), 
                 padding=(1, 1, 1), bias=False):
        super().__init__()
        
        # Handle different input formats
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        
        # Depthwise convolution (spatial + temporal)
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: each channel processed separately
            bias=False
        )
        
        # Batch norm after depthwise
        self.bn_depthwise = nn.BatchNorm3d(in_channels)
        
        # Pointwise convolution (1x1x1 - combines channels)
        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=bias
        )
        
        # Final batch norm
        self.bn_pointwise = nn.BatchNorm3d(out_channels)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for depthwise separable convolutions"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = F.relu(x)
        
        # Pointwise convolution  
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        x = F.relu(x)
        
        return x

class EfficientRCL_block_3D(nn.Module):
    """
    Memory and compute efficient version using depthwise separable convolutions
    Standalone implementation for better control and efficiency
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(3, 3, 3), t_steps=3, 
                 temporal_stride=1, dropout_rate=0.1):
        super(EfficientRCL_block_3D, self).__init__()
        
        self.t_steps = t_steps
        self.out_channels = out_channels
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2, (kernel_size[2]-1)//2)
        
        # Efficient depthwise separable convolutions
        self.feed_forward_conv = DepthwiseSeparable3D(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=(temporal_stride, 1, 1),
            padding=padding
        )
        
        self.recurrent_conv = DepthwiseSeparable3D(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=padding
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through efficient RCL block
        
        Args:
            x: Input tensor (batch_size, channels, time, height, width)
            
        Returns:
            Refined features after recurrent processing
        """
        # Step 1: Feed-forward pass (includes BN and ReLU in DepthwiseSeparable3D)
        x_ff = self.feed_forward_conv(x)
        
        # Step 2: Initialize recurrent state  
        x_recurrent = x_ff  # Already activated in DepthwiseSeparable3D
        
        # Step 3: Recurrent refinement loop
        for t in range(self.t_steps):
            # Residual connection: combine current state with original features
            recurrent_input = x_recurrent + x_ff
            
            # Process through recurrent convolution (includes BN and ReLU)
            x_recurrent = self.recurrent_conv(recurrent_input)
            
            # Apply dropout (only during training)
            if self.training:
                x_recurrent = self.dropout(x_recurrent)
        
        return x_recurrent

# Utility function to choose block type
def get_rcl_block(efficient=False):
    """
    Factory function to get appropriate RCL block
    
    Args:
        efficient: If True, return EfficientRCL_block_3D, else RCL_block_3D
    
    Returns:
        RCL block class
    """
    return EfficientRCL_block_3D if efficient else RCL_block_3D
