import torch
import numpy as np
import matplotlib.pyplot as plt
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent, device

def run_simulation(agent, use_hat_in_actor):
    """Runs one full simulation with a given agent and returns history."""
    env = PllEnv()
    state = env.reset()
    done = False
    
    freq_history = [env.vco_freq / 1e9]
    phase_error_history = [env.phase_error]
    time_history = [env.time * 1e6]

    agent.actor.eval() # Set actor to evaluation mode

    while not done:
        # Pass the use_hat flag correctly during evaluation
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = agent.actor(state_tensor, use_hat=use_hat_in_actor).cpu().data.numpy().flatten()

        state, _, done, _ = env.step(action)
        
        freq_history.append(env.vco_freq / 1e9)
        phase_error_history.append(env.phase_error)
        time_history.append(env.time * 1e6)
        
    return time_history, freq_history, phase_error_history

def simulate_static_pll():
    """Simulates a static (non-learning) PLL which should now fail realistically."""
    env = PllEnv()
    env.reset()
    done = False
    
    freq_history = [env.vco_freq / 1e9]
    phase_error_history = [env.phase_error]
    time_history = [env.time * 1e6]

    while not done:
        # A static PLL has a fixed control loop; equivalent to zero action in our simplified model.
        action = np.zeros(env._get_state().shape[0]) # Action dim should match state
        # In a more complex model, this would be a fixed PI controller.
        # For simplicity, we keep action=0 to show its inability to adapt.
        state, _, done, _ = env.step(action)
        freq_history.append(env.vco_freq / 1e9)
        phase_error_history.append(env.phase_error)
        time_history.append(env.time * 1e6)
        
    return time_history, freq_history, phase_error_history


def plot_results(results):
    """Plots the final results in a format similar to the original paper."""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # --- Plot Frequency Response ---
    ax1.plot(results["static"]["time"], results["static"]["freq"], 'gray', linestyle='--', label='STATIC_PLL', alpha=0.8)
    # The 'ideal' plot now shows a HAT-trained policy in an ideal (no-noise) execution environment
    ax1.plot(results["ideal"]["time"], results["ideal"]["freq"], 'm-.', label='RICC_IDEAL (HAT Policy)', alpha=0.9)
    ax1.plot(results["hat"]["time"], results["hat"]["freq"], 'b-', label='RICC_HAT', linewidth=2.5)
    
    ax1.set_title('(a) Instantaneous VCO Frequency', fontsize=16)
    ax1.set_ylabel('Frequency [GHz]', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim(-1, 9)

    # --- Plot Phase Error ---
    ax2.plot(results["static"]["time"], results["static"]["phase"],'gray', linestyle='--', label='STATIC_PLL', alpha=0.8)
    ax2.plot(results["ideal"]["time"], results["ideal"]["phase"], 'm-.', label='RICC_IDEAL (HAT Policy)', alpha=0.9)
    ax2.plot(results["hat"]["time"], results["hat"]["phase"], 'b-', label='RICC_HAT', linewidth=2.5)

    ax2.set_title('(b) Phase Error', fontsize=16)
    ax2.set_xlabel('Time [μs]', fontsize=14)
    ax2.set_ylabel('Phase Error [rad]', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(-8, 8)
    
    plt.tight_layout()
    plt.savefig("adversarial_stress_test_realistic.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # You must have run main.py to generate a trained model first.
    
    # Redefine dimensions based on new environment state
    env = PllEnv()
    state_dim = env.reset().shape[0]
    action_dim = 2 # Action space is now smaller for simplicity

    results = {}
    
    # Load the single HAT-trained agent
    agent = TD3_HAT_Agent(state_dim, action_dim, use_hat=True)
    try:
        agent.load("models/ricc_hat_final")
        print("Successfully loaded pre-trained model.")
    except FileNotFoundError:
        print("ERROR: Could not find pre-trained model 'models/ricc_hat_final_actor.pth'.")
        print("Please run main.py to train and save a model first.")
        exit()

    print("Simulating RICC-HAT agent...")
    t_h, f_h, p_h = run_simulation(agent, use_hat_in_actor=True)
    results["hat"] = {"time": t_h, "freq": f_h, "phase": p_h}

    # Here, 'RICC_IDEAL' means we run the *same HAT-trained agent* but turn off the noise
    # in the actor's forward pass. This shows how the learned policy performs in a 'perfect' hardware environment.
    print("Simulating RICC_IDEAL (HAT policy on ideal hardware)...")
    t_i, f_i, p_i = run_simulation(agent, use_hat_in_actor=False)
    results["ideal"] = {"time": t_i, "freq": f_i, "phase": p_i}
    
    print("Simulating STATIC PLL...")
    t_s, f_s, p_s = simulate_static_pll()
    results["static"] = {"time": t_s, "freq": f_s, "phase": p_s}

    print("Plotting results...")
    plot_results(results)
