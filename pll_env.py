import numpy as np

class PllEnv:
    def __init__(self, dt=1e-7):
        self.dt = dt
        self.target_freq = 8e9
        
        # State variables
        self.vco_freq = self.target_freq
        self.phase_error = 0.0
        self.v_ctrl = 1.0
        
        # PLL parameters
        self.kvco = 1.8e9
        self.icp = 500e-6
        self.loop_gain = 1.0
        
        self.time = 0.0

        # --------- NEW: REWARD SCALING FACTOR ---------
        # This constant is used to scale the raw reward into a more stable range.
        self.reward_scale = 1e6 
        # ----------------------------------------------

    def reset(self):
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
        # A simplified but representative state vector
        return np.array([self.phase_error, self.v_ctrl, (self.vco_freq - self.target_freq) / 1e9])

    def step(self, action):
        self.time += self.dt
        
        self.icp *= (1 + action[0] * 0.1)
        self.loop_gain *= (1 + action[1] * 0.1)
        
        if 35e-6 <= self.time < 35e-6 + self.dt:
            self.target_freq = 200e6
        if 80e-6 <= self.time < 80e-6 + self.dt:
            self.kvco *= 1.35
            self.icp *= 0.80
        
        self.phase_error += (self.vco_freq - self.target_freq) * 2 * np.pi * self.dt
        
        error_correction = -self.phase_error * self.loop_gain * self.icp
        self.v_ctrl += error_correction * self.dt
        self.v_ctrl = np.clip(self.v_ctrl, 0, 2.0)
        
        self.vco_freq = self.target_freq + (self.v_ctrl - 1.0) * self.kvco

        # --- MODIFIED: Calculate and Scale the Reward ---
        # Heavily penalize large phase errors to encourage stability.
        # We use a quadratic penalty which is common in control tasks.
        raw_reward = -(self.phase_error**2)
        
        # Normalize/Scale the reward to a more stable range for the neural network.
        scaled_reward = raw_reward / self.reward_scale
        # ----------------------------------------------------
        
        done = self.time > 120e-6
        
        return self._get_state(), scaled_reward, done, {}
