"""
BitLinear Model Implementation for Agent Forge
Implements models with ternary weights (-1, 0, 1) for efficient neural network training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class BitLinear(nn.Module):
    """
    BitLinear layer with ternary weights (-1, 0, 1)
    Based on the BitNet architecture for efficient 1-bit transformers
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Quantization scale factors
        self.weight_scale = nn.Parameter(torch.ones(1))

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with ternary-friendly distribution"""
        # Initialize to encourage ternary values
        nn.init.uniform_(self.weight, -1.5, 1.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weight(self) -> torch.Tensor:
        """
        Quantize weights to ternary values (-1, 0, 1)
        Using sign-based quantization with threshold
        """
        # Compute threshold for sparsity
        threshold = 0.7 * torch.abs(self.weight).mean()

        # Quantize: values below threshold become 0, others become -1 or 1
        weight_sign = torch.sign(self.weight)
        weight_mask = (torch.abs(self.weight) > threshold).float()
        quantized = weight_sign * weight_mask

        return quantized * self.weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights"""
        # Get quantized weights
        weight_q = self.quantize_weight()

        # Linear transformation
        output = F.linear(x, weight_q, self.bias)

        return output

    def get_weight_distribution(self) -> dict:
        """Get distribution of quantized weights"""
        with torch.no_grad():
            quantized = self.quantize_weight() / self.weight_scale
            return {
                'negative_ones': (quantized == -1).sum().item(),
                'zeros': (quantized == 0).sum().item(),
                'positive_ones': (quantized == 1).sum().item(),
                'total': quantized.numel()
            }


class BitLinearModel(nn.Module):
    """
    Complete model using BitLinear layers
    Suitable for Phase 5 sophisticated training with ternary weights
    """

    def __init__(self,
                 input_dim: int = 784,
                 hidden_dims: list = [256, 128, 64],
                 output_dim: int = 10,
                 activation: str = 'relu'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(BitLinear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())

            # Add normalization for stability
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        layers.append(BitLinear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.model(x)

    def get_weight_statistics(self) -> dict:
        """Get statistics about all BitLinear layers"""
        stats = {
            'layers': {},
            'total': {'negative_ones': 0, 'zeros': 0, 'positive_ones': 0, 'total': 0}
        }

        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                dist = module.get_weight_distribution()
                stats['layers'][name] = dist
                stats['total']['negative_ones'] += dist['negative_ones']
                stats['total']['zeros'] += dist['zeros']
                stats['total']['positive_ones'] += dist['positive_ones']
                stats['total']['total'] += dist['total']

        # Calculate percentages
        total = stats['total']['total']
        if total > 0:
            stats['total']['percentages'] = {
                'negative_ones': 100 * stats['total']['negative_ones'] / total,
                'zeros': 100 * stats['total']['zeros'] / total,
                'positive_ones': 100 * stats['total']['positive_ones'] / total
            }

        return stats

    def extract_ternary_weights(self) -> dict:
        """
        Extract all weights in ternary form for visualization
        Returns dictionary mapping layer names to quantized weight tensors
        """
        weights = {}

        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                with torch.no_grad():
                    # Get quantized weights normalized to -1, 0, 1
                    quantized = module.quantize_weight() / module.weight_scale
                    weights[name] = quantized.cpu().numpy()

        return weights


class BitLinearTrainer:
    """
    Specialized trainer for BitLinear models with Phase 5 features
    """

    def __init__(self, model: BitLinearModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device

        # Training metrics
        self.training_history = []
        self.weight_evolution = []

    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                  optimizer: torch.optim.Optimizer,
                  criterion: nn.Module) -> float:
        """Single training step with ternary weight awareness"""
        self.model.train()

        # Forward pass
        x, y = x.to(self.device), y.to(self.device)
        outputs = self.model(x)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability with ternary weights
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        return loss.item()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Evaluate model accuracy and loss"""
        self.model.eval()

        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)

            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y).item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)

        return loss, accuracy

    def get_weight_snapshot(self) -> dict:
        """Get current weight distribution snapshot"""
        stats = self.model.get_weight_statistics()
        ternary_weights = self.model.extract_ternary_weights()

        return {
            'statistics': stats,
            'weights': ternary_weights,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }


def create_bitlinear_model_for_phase5(
    input_dim: int = 784,
    complexity_level: int = 5  # 1-10 scale from Phase 5
) -> BitLinearModel:
    """
    Create a BitLinear model configured for Phase 5 training
    Complexity scales with the level (1-10)
    """

    # Scale architecture with complexity level
    base_hidden = 64
    hidden_dims = []

    if complexity_level <= 3:
        # Simple architecture for levels 1-3
        hidden_dims = [base_hidden * 2]
    elif complexity_level <= 6:
        # Medium architecture for levels 4-6
        hidden_dims = [base_hidden * 4, base_hidden * 2]
    elif complexity_level <= 8:
        # Complex architecture for levels 7-8
        hidden_dims = [base_hidden * 8, base_hidden * 4, base_hidden * 2]
    else:
        # Very complex architecture for levels 9-10
        hidden_dims = [base_hidden * 16, base_hidden * 8, base_hidden * 4, base_hidden * 2]

    # Create model
    model = BitLinearModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=10,
        activation='gelu' if complexity_level > 5 else 'relu'
    )

    print(f"Created BitLinear model for Phase 5 Level {complexity_level}")
    print(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 10")

    # Get initial weight distribution
    stats = model.get_weight_statistics()
    print(f"Initial weight distribution: -1: {stats['total'].get('percentages', {}).get('negative_ones', 0):.1f}%, "
          f"0: {stats['total'].get('percentages', {}).get('zeros', 0):.1f}%, "
          f"1: {stats['total'].get('percentages', {}).get('positive_ones', 0):.1f}%")

    return model


if __name__ == "__main__":
    # Test the BitLinear model
    model = create_bitlinear_model_for_phase5(input_dim=784, complexity_level=5)

    # Test forward pass
    x = torch.randn(32, 784)
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Extract ternary weights
    weights = model.extract_ternary_weights()
    for name, weight in weights.items():
        unique_values = np.unique(weight)
        print(f"Layer {name}: shape={weight.shape}, unique values={unique_values}")