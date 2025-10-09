import numpy as np

class PllEnv:
    """
    An improved, more realistic behavioral model of a Phase-Locked Loop (PLL).
    This version includes second-order dynamics to simulate overshoot and settling time.
    """
    def __init__(self, dt=1e-8): # <--- Using a smaller timestep for better resolution
        self.dt = dt
        self.base_kvco = 1.8e9
        self.base_icp = 500e-6
        self.reward_scale = 1e6
        
        # --- NEW: Second-Order Dynamics State Variables ---
        # These model the state of the loop filter, giving it "inertia"
        self.lf_integral_state = 0.0  # Represents the charge on the main integrating capacitor
        self.lf_prop_state = 0.0    # Represents the charge on the proportional path capacitor
        # ----------------------------------------------------

        # Initialize other state variables
        self.reset()
        
    def reset(self):
        self.target_freq = 8e9
        self.vco_phase = 0.0
        self.ref_phase = 0.0
        self.phase_error = 0.0
        self.v_ctrl = 1.0
        self.time = 0.0
        
        self.kvco = self.base_kvco
        self.icp = self.base_icp
        
        # Reset new state variables
        self.lf_integral_state = self.v_ctrl
        self.lf_prop_state = 0.0
        self.vco_freq = self.target_freq

        return self._get_state()

    def _get_state(self):
        return np.array([
            np.sin(self.phase_error), # Use sin/cos of phase error, common in practice
            np.cos(self.phase_error),
            self.v_ctrl,
        ], dtype=np.float32)

    def step(self, action):
        self.time += self.dt

        # --- 1. Agent Action and PVT Drift (no changes here) ---
        current_icp = self.icp * (1 + action[0] * 0.1)
        # We can simplify the action space for now to make learning easier
        # action[1] can adjust a damping factor (zeta) for the loop
        
        if 35e-6 <= self.time < 35e-6 + self.dt:
            self.target_freq = 200e6
        if 80e-6 <= self.time < 80e-6 + self.dt:
            self.kvco *= 1.35
            current_icp *= 0.80
        
        # --- 2. More Realistic PLL Dynamics ---
        # Phase detector output is proportional to phase error
        pfd_current = current_icp * (self.phase_error / (2 * np.pi))

        # Loop filter dynamics (models a simplified pole-zero response)
        # Proportional path (creates the zero for stability)
        self.lf_prop_state = pfd_current * 5e3 # R_p
        # Integral path (creates the pole at origin for tracking)
        self.lf_integral_state += pfd_current / 47e-12 * self.dt # C_p

        # Control voltage is the sum of the two paths
        self.v_ctrl = self.lf_integral_state + self.lf_prop_state
        self.v_ctrl = np.clip(self.v_ctrl, 0, 1.8)

        # VCO frequency now changes based on control voltage
        self.vco_freq = (self.v_ctrl) * self.kvco

        # Update phase based on frequency
        self.vco_phase += 2 * np.pi * self.vco_freq * self.dt
        self.ref_phase += 2 * np.pi * self.target_freq * self.dt
        self.phase_error = (self.ref_phase - self.vco_phase) % (2*np.pi)
        if self.phase_error > np.pi:
             self.phase_error -= 2*np.pi

        # --- 3. Reward (no changes here) ---
        raw_reward = -(self.phase_error**2)
        scaled_reward = raw_reward / 1.0 # Smaller scale is better now
        
        done = self.time > 120e-6
        return self._get_state(), scaled_reward, done, {}
