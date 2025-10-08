import numpy as np

class PllEnv:
    """
    A simplified behavioral model of a Phase-Locked Loop (PLL) environment.
    This environment simulates the core dynamics for the reinforcement learning agent,
    including the adversarial stress test scenario described in the paper.
    """
    def __init__(self, dt=1e-7):
        # Simulation parameters
        self.dt = dt  # Simulation timestep (100 ns)
        
        # --- Base PLL Parameters ---
        # These are the nominal 'TT corner' values
        self.base_kvco = 1.8e9   # VCO gain (Hz/V)
        self.base_icp = 500e-6   # Charge pump current (A)

        # Reward scaling factor to stabilize learning
        self.reward_scale = 1e6 
        
        # Initialize state variables
        self.target_freq = 0
        self.vco_freq = 0
        self.phase_error = 0.0
        self.v_ctrl = 0.0
        self.time = 0.0
        # Dynamic parameters that change during simulation
        self.kvco = 0
        self.icp = 0
        self.loop_gain = 0
        
    def reset(self):
        """Resets the environment to the initial locked state."""
        self.target_freq = 8e9
        self.vco_freq = self.target_freq
        self.phase_error = 0.0
        self.v_ctrl = 1.0 # Normalized control voltage starts at midpoint
        self.time = 0.0
        
        # Reset dynamic parameters to their base values
        self.kvco = self.base_kvco
        self.icp = self.base_icp
        self.loop_gain = 1.0
        
        return self._get_state()

    def _get_state(self):
        """Assembles and returns the current state vector for the agent."""
        # This simplified state vector captures the most critical dynamics
        state = np.array([
            self.phase_error, 
            self.v_ctrl, 
            (self.vco_freq - self.target_freq) / 1e9 # Frequency error in GHz
        ], dtype=np.float32)
        return state

    def step(self, action):
        """Executes one time step in the environment."""
        self.time += self.dt

        # --- 1. Apply Agent's Action ---
        # The agent's actions are small, continuous adjustments to the loop parameters
        self.icp *= (1 + action[0] * 0.1) # Action modifies Icp by up to +/- 10%
        self.loop_gain *= (1 + action[1] * 0.1) # Action modifies loop gain/damping
        
        # --- 2. Simulate Adversarial Events ---
        if 35e-6 <= self.time < 35e-6 + self.dt:
            self.target_freq = 200e6 # N-Switch event: large frequency jump
        if 80e-6 <= self.time < 80e-6 + self.dt:
            self.kvco *= 1.35  # Unforeseen PVT drift: K_vco increases by 35%
            self.icp *= 0.80   # Unforeseen PVT drift: I_cp decreases by 20%
        
        # --- 3. Simulate PLL Dynamics (Simplified 2nd Order Model) ---
        self.phase_error += (self.vco_freq - self.target_freq) * 2 * np.pi * self.dt
        
        # A PI controller approximates a charge-pump PLL's loop filter
        error_correction = -self.phase_error * self.loop_gain * self.icp
        self.v_ctrl += error_correction * self.dt
        self.v_ctrl = np.clip(self.v_ctrl, 0, 2.0)
        
        self.vco_freq = self.target_freq + (self.v_ctrl - 1.0) * self.kvco

        # --- 4. Calculate Reward ---
        # The reward function heavily penalizes phase error to encourage stability.
        raw_reward = -(self.phase_error**2)
        scaled_reward = raw_reward / self.reward_scale # Scale for learning stability
        
        done = self.time > 120e-6
        
        return self._get_state(), scaled_reward, done, {}
