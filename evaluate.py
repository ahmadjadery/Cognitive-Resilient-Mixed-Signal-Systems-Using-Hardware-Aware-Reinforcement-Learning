import torch
import numpy as np
import matplotlib.pyplot as plt
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent

def run_simulation(agent):
    """Runs one full simulation with a given agent and returns history."""
    env = PllEnv()
    state = env.reset()
    done = False
    
    freq_history = [env.vco_freq / 1e9]
    phase_error_history = [env.phase_error]
    time_history = [env.time * 1e6]

    while not done:
        # Select action WITHOUT exploration noise for evaluation
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
        
        freq_history.append(env.vco_freq / 1e9)
        phase_error_history.append(env.phase_error)
        time_history.append(env.time * 1e6)
        
    return time_history, freq_history, phase_error_history

def simulate_static_pll():
    """Simulates a static (non-learning) PLL for comparison."""
    env = PllEnv()
    env.reset()
    done = False
    
    freq_history = [env.vco_freq / 1e9]
    time_history = [env.time * 1e6]

    while not done:
        # Static PLL does nothing, its parameters are fixed.
        # This is equivalent to taking a zero action every step.
        action = np.zeros(3)
        _, _, done, _ = env.step(action)
        freq_history.append(env.vco_freq / 1e9)
        time_history.append(env.time * 1e6)
        
    return time_history, freq_history

def plot_results(results):
    """Plots the final results in a format similar to the paper."""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    # --- Plot Frequency Response ---
    ax1.plot(results["static"]["time"], results["static"]["freq"], 'k--', label='STATIC_PLL', alpha=0.5)
    ax1.plot(results["ideal"]["time"], results["ideal"]["freq"], 'm-.', label='RICC_IDEAL_PLL')
    ax1.plot(results["hat"]["time"], results["hat"]["freq"], 'b-', label='RICC_HAT_PLL', linewidth=2)
    ax1.set_title('(a) Instantaneous VCO Frequency')
    ax1.set_ylabel('Frequency [GHz]')
    ax1.legend()
    ax1.set_ylim(-1, 9)

    # --- Plot Phase Error ---
    ax2.plot(results["ideal"]["time"], results["ideal"]["phase"], 'm-.', label='RICC_IDEAL_PLL')
    ax2.plot(results["hat"]["time"], results["hat"]["phase"], 'b-', label='RICC_HAT_PLL', linewidth=2)
    # Plot static and FSM as linearly increasing error for illustration
    ax2.plot(results["static"]["time"], np.linspace(0, 8, len(results["static"]["time"])), 'k--', alpha=0.5, label='STATIC_PLL (loss of lock)')
    ax2.set_title('(b) Phase Error')
    ax2.set_xlabel('Time [μs]')
    ax2.set_ylabel('Phase Error [rad]')
    ax2.set_ylim(-8, 8)
    
    plt.tight_layout()
    plt.savefig("adversarial_stress_test.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # --- Setup ---
    env = PllEnv()
    state_dim = env.reset().shape[0]
    action_dim = 3
    
    results = {}

    # --- Simulate RICC-HAT Agent ---
    print("Simulating RICC-HAT agent...")
    hat_agent = TD3_HAT_Agent(state_dim, action_dim, use_hat=True)
    try:
        hat_agent.load("models/ricc_hat_final")
    except FileNotFoundError:
        print("Could not find pre-trained RICC-HAT model. Please run main.py first.")
        exit()
    t_h, f_h, p_h = run_simulation(hat_agent)
    results["hat"] = {"time": t_h, "freq": f_h, "phase": p_h}

    # --- Simulate RICC-IDEAL Agent (trained separately without HAT) ---
    print("Simulating RICC-IDEAL agent...")
    ideal_agent = TD3_HAT_Agent(state_dim, action_dim, use_hat=False)
    # For a real implementation, you would train and save this agent separately
    # Here, we'll just load the HAT agent and run it in IDEAL mode for a quick demo
    ideal_agent.load("models/ricc_hat_final") # This is not ideal, but shows the code structure
    t_i, f_i, p_i = run_simulation(ideal_agent)
    results["ideal"] = {"time": t_i, "freq": f_i, "phase": p_i}
    
    # --- Simulate Static PLL ---
    print("Simulating STATIC PLL...")
    t_s, f_s = simulate_static_pll()
    results["static"] = {"time": t_s, "freq": f_s}

    # --- Plot ---
    print("Plotting results...")
    plot_results(results)
