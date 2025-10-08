import torch
import torch.nn as nn
import numpy as np

# This import is needed to access the globally defined device
from utils import device

def stochastic_forward_pass(weights, inputs):
    """
    Simulates a forward pass through one layer of a ReRAM-based AIMC core.

    This function is the core of the HAT methodology. It simulates the
    physical computation of a matrix-vector multiplication on a non-ideal
    ReRAM crossbar array by injecting statistically-modeled hardware noise.

    Args:
        weights (torch.Tensor): The ideal weight matrix of the neural network layer.
        inputs (torch.Tensor): The input vector to the layer (already on the correct device).

    Returns:
        torch.Tensor: The quantized, noisy output vector on the same device.
    """
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    # 1. ReRAM Conductance Variation (Log-Normal Distribution)
    # Models device-to-device and cycle-to-cycle programming variability.
    # The parameters (mean=0, sigma=0.12) are based on device characterization data.
    g_variation_np = np.random.lognormal(mean=0, sigma=0.12, size=weights.shape)
    # CRITICAL FIX: Ensure the new tensor is created on the same device as the inputs.
    g_variation = torch.tensor(g_variation_np, dtype=torch.float32, device=inputs.device)
    noisy_weights = weights * g_variation

    # 2. Interconnect Parasitics (IR Drop Model)
    # A simplified model for voltage drop across resistive word lines.
    ir_drop_factor = 0.99 
    effective_inputs = inputs * ir_drop_factor

    # 3. Readout Noise and Quantization
    # Performs the dot product (MVM) using the noisy, degraded components.
    if len(inputs.shape) == 1: # Handle single vector case
        output = torch.matmul(effective_inputs, noisy_weights.T)
    else: # Handle batch of vectors case
        output = torch.matmul(effective_inputs, noisy_weights.T)

    # Models thermal noise from Transimpedance Amplifiers (TIAs).
    thermal_noise = torch.randn_like(output) * 0.8e-6
    
    # Simulates the finite 8-bit resolution of the SAR ADCs.
    quantization_levels = 2**8
    quantized_output = torch.round((output + thermal_noise) * (quantization_levels / 2)) / (quantization_levels / 2)
    
    # Clip to ensure output is within the ADC's output range [-1, 1].
    quantized_output = torch.clamp(quantized_output, -1.0, 1.0)
    
    return quantized_output

class Actor(nn.Module):
    """
    The Actor network, which represents the control policy.
    It maps a state to a deterministic action.
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 32)
        self.layer_2 = nn.Linear(32, 24)
        self.layer_3 = nn.Linear(24, 16)
        self.layer_4 = nn.Linear(16, action_dim)
        
    def forward(self, x, use_hat=True):
        """Defines the forward pass of the Actor."""
        if use_hat:
            # HAT mode: Use the noisy, physically-grounded forward pass.
            x = stochastic_forward_pass(self.layer_1.weight, x)
            x = torch.relu(x)
            x = stochastic_forward_pass(self.layer_2.weight, x)
            x = torch.relu(x)
            x = stochastic_forward_pass(self.layer_3.weight, x)
            x = torch.relu(x)
            x = stochastic_forward_pass(self.layer_4.weight, x)
        else:
            # Ideal mode: Use a standard, noiseless forward pass for comparison.
            x = torch.relu(self.layer_1(x))
            x = torch.relu(self.layer_2(x))
            x = torch.relu(self.layer_3(x))
            x = self.layer_4(x)
            
        # The final activation function, conceptually performed by the DCNU's NLAU.
        return torch.tanh(x)

class Critic(nn.Module):
    """
    The Critic network (Q-function), which estimates the value of a state-action pair.
    Used by the agent to learn and improve its policy.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Input is the state and action combined.
        self.layer_1 = nn.Linear(state_dim + action_dim, 32)
        self.layer_2 = nn.Linear(32, 24)
        self.layer_3 = nn.Linear(24, 1) # Outputs a single scalar Q-value.

    def forward(self, state, action):
        """Defines the forward pass of the Critic."""
        x = torch.cat([state, action], 1)
        # For this study, the Critic is assumed to run on an ideal digital processor.
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        return self.layer_3(x)
