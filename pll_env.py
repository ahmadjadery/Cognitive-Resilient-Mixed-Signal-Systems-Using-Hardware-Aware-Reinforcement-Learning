import numpy as np

class PllEnv:
    """
    A simplified behavioral model of a Phase-Locked Loop (PLL) environment.
    This environment simulates the core dynamics for the reinforcement learning agent.
    """
    def __init__(self, dt=1e-7):
        self.dt = dt  # Simulation timestep (100 ns)
        self.target_freq = 8e9  # Target frequency in Hz (8 GHz)
        
        # State variables
        self.vco_freq = self.target_freq
        self.phase_error = 0.0
        self.v_ctrl = 1.0  # Normalized control voltage
        
        # PLL parameters (these are affected by the agent's actions)
        self.kvco = 1.8e9  # VCO gain (GHz/V) - Base value
        self.icp = 500e-6  # Charge pump current - Base value
        self.loop_gain = 1.0 # Combined loop gain factor
        
        self.time = 0.0

    def reset(self):
        """Resets the environment to its initial state."""
        self.target_freq = 8e9
        self.vco_freq = self.target_freq
        self.phase_error = 0.0
        self.v_ctrl = 1.0
        self.kvco = 1.8e9
        self.icp = 500e-6
        self.loop_gain = 1.0
        self.time = 0.0
        return self._get_state()

    def _get_state(self):
        """Returns the current state vector for the agent."""
        # This is a simplified 13-dimensional state vector from the paper.
        # Here we use a smaller, representative state.
        return np.array([self.phase_error, self.v_ctrl, (self.vco_freq - self.target_freq) / 1e9])

    def step(self, action):
        """
        Executes one time step in the environment.

        Args:
            action (np.array): A 3-element action vector {delta_icp, delta_kp, delta_ki}
                               from the RL agent, normalized between -1 and 1.
        """
        # --- Apply agent's action ---
        # Action modifies the loop parameters
        self.icp *= (1 + action[0] * 0.1) # Action modifies Icp by up to 10%
        self.loop_gain *= (1 + action[1] * 0.1) # Actions modify loop gain/damping

        # --- Simulate Adversarial Events ---
        self.time += self.dt
        if 35e-6 <= self.time < 35e-6 + self.dt:
            # 1. N-Switch event: 40x frequency jump down
            self.target_freq = 200e6
        if 80e-6 <= self.time < 80e-6 + self.dt:
            # 2. Unforeseen PVT drift
            self.kvco *= 1.35  # K_vco +35%
            self.icp *= 0.80   # I_cp -20%
        
        # --- PLL Dynamics (Simplified 2nd Order Model) ---
        # 1. Phase error accumulates based on frequency difference
        self.phase_error += (self.vco_freq - self.target_freq) * 2 * np.pi * self.dt
        
        # 2. Control voltage adjusts to correct phase error
        # A PI controller is a good behavioral approximation for a charge-pump PLL
        error_correction = -self.phase_error * self.loop_gain * self.icp
        self.v_ctrl += error_correction * self.dt
        self.v_ctrl = np.clip(self.v_ctrl, 0, 2.0) # Clip control voltage
        
        # 3. VCO frequency responds to new control voltage
        self.vco_freq = self.target_freq + (self.v_ctrl - 1.0) * self.kvco

        # --- Calculate Reward ---
        # Reward is high when phase error is low, heavily penalizes large errors
        reward = -np.abs(self.phase_error) - (self.phase_error**2) * 0.1
        
        done = self.time > 120e-6 # End of episode
        
        return self._get_state(), reward, done, {}
