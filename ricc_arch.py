import torch
import torch.nn as nn
import numpy as np

# This function is the core of the HAT methodology.
# It simulates the physical computation of a matrix-vector multiplication
# on a non-ideal ReRAM crossbar array.

def stochastic_forward_pass(weights, inputs):
    
    Simulates a forward pass through one layer of a ReRAM-based AIMC core.

    Args:
        weights (torch.Tensor): The ideal weight matrix of the neural network layer.
        inputs (torch.Tensor): The input vector to the layer.

    Returns:
        torch.Tensor: The quantized, noisy output vector.
    
    # Ensure inputs are tensors for torch operations
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)

    # 1. ReRAM Conductance Variation (Log-Normal Distribution)
    # This models device-to-device and cycle-to-cycle programming variability.
    # The parameters (mean=0, sigma=0.12) are based on device characterization.
    # Note: we use numpy here as lognormal is more direct, then convert to tensor.
    g_variation_np = np.random.lognormal(mean=0, sigma=0.12, size=weights.shape)
    g_variation = torch.tensor(g_variation_np, dtype=torch.float32)
    noisy_weights = weights * g_variation

    # 2. Interconnect Parasitics (IR Drop Model)
    # This models the voltage drop across resistive word lines (rows).
    # A simplified model where drop is proportional to input and line resistance.
    # r_line = 50 ohms per unit length (conceptual)
    # For simplicity, we'll model this as a small, systematic input degradation.
    # A full physical model (calculate_ir_drop) would be more complex.
    ir_drop_factor = 0.99 # A simplified factor to represent minor voltage loss
    effective_inputs = inputs * ir_drop_factor

    # 3. Readout Noise and Quantization
    # Performs the dot product using the noisy weights and degraded inputs.
    output = torch.matmul(effective_inputs, noisy_weights.T) # Using matrix multiplication convention
    
    # Models thermal noise from TIAs (Transimpedance Amplifiers)
    thermal_noise = torch.randn_like(output) * 0.8e-6 # Scaled appropriately for this domain
    
    # Simulates the finite 8-bit resolution of the SAR ADCs.
    # Assumes output range is normalized between -1 and 1.
    quantization_levels = 2**8
    quantized_output = torch.round((output + thermal_noise) * (quantization_levels / 2)) / (quantization_levels / 2)
    
    # Clip to ensure output is within the ADC range
    quantized_output = torch.clamp(quantized_output, -1.0, 1.0)
    
    return quantized_output

# --- Define Actor-Critic Architectures using PyTorch ---

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # In a real implementation, these layers would be mapped to crossbar arrays.
        # Here we define them as standard linear layers.
        self.layer_1 = nn.Linear(state_dim, 32)
        self.layer_2 = nn.Linear(32, 24)
        self.layer_3 = nn.Linear(24, 16)
        self.layer_4 = nn.Linear(16, action_dim)
        
    def forward(self, x, use_hat=True):
        if use_hat:
            # Apply the stochastic forward pass for each layer
            x = stochastic_forward_pass(self.layer_1.weight, x)
            # The original paper implies activation functions are applied by the DCNU
            x = torch.relu(x)
            x = stochastic_forward_pass(self.layer_2.weight, x)
            x = torch.relu(x)
            x = stochastic_forward_pass(self.layer_3.weight, x)
            x = torch.relu(x)
            x = stochastic_forward_pass(self.layer_4.weight, x)
        else: # Ideal pass for RICC-IDEAL comparison
            x = torch.relu(self.layer_1(x))
            x = torch.relu(self.layer_2(x))
            x = torch.relu(self.layer_3(x))
            x = self.layer_4(x)
            
        # The final tanh activation, performed by the NLAU in the DCNU
        return torch.tanh(x)


class Critic(nn.Module):
    
    The Critic network (Q-function) for the TD3 algorithm.
    It takes a state and an action and outputs a single Q-value.
    In TD3, two Critic networks are used to reduce overestimation.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # The input to the first layer is the state and action concatenated
        self.layer_1 = nn.Linear(state_dim + action_dim, 32)
        self.layer_2 = nn.Linear(32, 24)
        self.layer_3 = nn.Linear(24, 1) # Outputs a single Q-value

    def forward(self, state, action):
        # Concatenate state and action to form the input
        x = torch.cat([state, action], 1)
        
        # In this implementation, for simplicity, we assume the critic runs on an
        # ideal digital co-processor and does not use the stochastic_forward_pass.
        # A more complex model could also apply HAT to the critic.
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        return self.layer_3(x)
