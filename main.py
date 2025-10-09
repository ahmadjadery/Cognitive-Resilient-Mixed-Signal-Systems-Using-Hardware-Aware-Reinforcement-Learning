import os
import torch
import numpy as np
from tqdm import tqdm
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent, ReplayBuffer

def main():
    # --- Hyperparameters ---
    MAX_EPISODES = 500
    START_TRAINING_STEPS = 5000  # Start training after this many steps in the buffer
    EXPLORATION_NOISE = 0.2     # A slightly higher noise can help with harder envs
    BATCH_SIZE = 256
    
    # --- Setup ---
    # 1. Initialize the new, more realistic environment
    env = PllEnv()
    
    # 2. CRITICAL UPDATE: Get state and action dimensions from the new environment
    state_dim = env.reset().shape[0] # This will now be 3
    # We simplified the action space to 2 for more stable learning
    action_dim = 2                   
    
    # 3. Initialize the Agent and Replay Buffer with the correct dimensions
    agent = TD3_HAT_Agent(state_dim=state_dim, action_dim=action_dim, use_hat=True)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
    
    # Create directory for saving models if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    print(f"Starting Training: State Dim: {state_dim}, Action Dim: {action_dim}")

    # --- Training Loop ---
    total_steps = 0
    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Use tqdm for a progress bar over steps within an episode for better feedback
        # Each episode has a fixed number of steps based on dt and end time
        episode_steps = int(120e-6 / env.dt)
        pbar = tqdm(range(episode_steps), desc=f"Episode {episode+1}/{MAX_EPISODES}")
        
        for step in pbar:
            total_steps += 1
            
            # 4. CRITICAL UPDATE: Ensure exploration noise matches the new action dimension
            action = (
                agent.select_action(state)
                + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
            ).clip(-1, 1)

            next_state, reward, done, _ = env.step(action)
            
            replay_buffer.add(state, action, next_state, reward, float(done))
            
            state = next_state
            episode_reward += reward

            # Start training only after the buffer has enough samples
            if total_steps > START_TRAINING_STEPS:
                agent.train(replay_buffer, BATCH_SIZE)
        
        # Print summary at the end of each episode
        print(f"Episode {episode+1} Complete. Total Steps: {total_steps}, Reward: {episode_reward:.2f}")

        # Save a snapshot of the model periodically
        if (episode + 1) % 50 == 0:
            print(f"--- Saving model at episode {episode+1} ---")
            agent.save(f"models/ricc_hat_episode_{episode+1}")


    # Save the final trained model
    agent.save("models/ricc_hat_final")
    print("Training complete. Final model saved to 'models/ricc_hat_final'.")

if __name__ == '__main__':
    main()
