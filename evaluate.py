import os
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Use the robust TkAgg backend to avoid display issues
import matplotlib.pyplot as plt
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent, device

# --- CONFIGURATION ---
# We will load the main checkpoint file saved during training.
CHECKPOINT_FILE_TO_LOAD = os.path.join("models", "checkpoint.pth")
# ---------------------

def run_simulation(agent, use_hat_in_actor):
    env = PllEnv()
    state = env.reset()
    done = False
    
    freq_history = [env.vco_freq / 1e9]
    phase_error_history = [env.phase_error]
    time_history = [env.time * 1e6]

    if agent:
        agent.actor.eval()

    with torch.no_grad():
        while not done:
            if agent:
                state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action = agent.actor(state_tensor, use_hat=use_hat_in_actor).cpu().data.numpy().flatten()
            else:
                action = np.zeros(2)
            
            state, _, done, _ = env.step(action)
            
            freq_history.append(env.vco_freq / 1e9)
            phase_error_history.append(env.phase_error)
            time_history.append(env.time * 1e6)
            
    # --- CRITICAL FIX: Return a dictionary, not a tuple ---
    return {
        "time": time_history,
        "freq": freq_history,
        "phase": phase_error_history
    }

def plot_results(results):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Plot Frequency Response
    ax1.plot(results["static"]["time"], results["static"]["freq"], 'gray', linestyle='--', label='STATIC_PLL', alpha=0.8)
    ax1.plot(results["ideal"]["time"], results["ideal"]["freq"], 'm-.', label='RICC_IDEAL (HAT Policy)', alpha=0.9)
    ax1.plot(results["hat"]["time"], results["hat"]["freq"], 'b-', label='RICC_HAT', linewidth=2.5)
    ax1.set_title('(a) Instantaneous VCO Frequency', fontsize=16)
    ax1.set_ylabel('Frequency [GHz]', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim(-1, 9)

    # Plot Phase Error
    ax2.plot(results["static"]["time"], results["static"]["phase"],'gray', linestyle='--', label='STATIC_PLL', alpha=0.8)
    ax2.plot(results["ideal"]["time"], results["ideal"]["phase"], 'm-.', label='RICC_IDEAL (HAT Policy)', alpha=0.9)
    ax2.plot(results["hat"]["time"], results["hat"]["phase"], 'b-', label='RICC_HAT', linewidth=2.5)
    ax2.set_title('(b) Phase Error', fontsize=16)
    ax2.set_xlabel('Time [μs]', fontsize=14)
    ax2.set_ylabel('Phase Error [rad]', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(-8, 8)
    
    plt.tight_layout()
    plt.savefig("adversarial_stress_test_result.png", dpi=300)
    print("\nPlot saved as 'adversarial_stress_test_result.png'")
    plt.show()

if __name__ == '__main__':
    results = {}
    
    # --- Load the trained agent ---
    env = PllEnv()
    state_dim = env.reset().shape[0]
    action_dim = 2 # Matches the latest main.py and pll_env.py
    
    agent = TD3_HAT_Agent(state_dim, action_dim, use_hat=True)
    try:
        # Tell the load function this is for evaluation only
        agent.load(CHECKPOINT_FILE_TO_LOAD, evaluate=True) 
        print(f"Successfully loaded model from checkpoint: {CHECKPOINT_FILE_TO_LOAD}")
    except FileNotFoundError:
        print(f"ERROR: Could not find checkpoint file '{CHECKPOINT_FILE_TO_LOAD}'.")
        print("Please run main.py to train and save a model first.")
        exit()

    # --- Run Simulations ---
    print("Simulating RICC-HAT (policy performance with hardware noise)...")
    results["hat"] = run_simulation(agent, use_hat_in_actor=True)

    print("Simulating RICC-IDEAL (policy performance on ideal hardware)...")
    results["ideal"] = run_simulation(agent, use_hat_in_actor=False)
    
    print("Simulating STATIC PLL...")
    # We pass 'None' for the agent, which tells run_simulation to use zero-action
    results["static"] = run_simulation(agent=None, use_hat_in_actor=False)

    # --- Plot Results ---
    print("Plotting results...")
    plot_results(results)
