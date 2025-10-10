import os
import torch
import numpy as np
from tqdm import tqdm
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent, ReplayBuffer

def main():
    # --- Hyperparameters ---
    MAX_EPISODES = 500
    START_TRAINING_STEPS = 5000
    EXPLORATION_NOISE = 0.2
    BATCH_SIZE = 256
    
    # --- Setup ---
    env = PllEnv()
    state_dim = env.reset().shape[0]
    action_dim = 2
    
    agent = TD3_HAT_Agent(state_dim=state_dim, action_dim=action_dim, use_hat=True)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    print(f"Starting Training: State Dim: {state_dim}, Action Dim: {action_dim}")

    # --- Training Loop ---
    total_steps = 0
    # The main progress bar is for the total number of episodes.
    for episode in tqdm(range(MAX_EPISODES), desc="Total Episodes"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            total_steps += 1
            
            # --- CRITICAL PERFORMANCE FIX ---
            with torch.no_grad():
                action = (
                    agent.select_action(state)
                    + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
                ).clip(-1, 1)
            # --------------------------------

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, float(done))
            state = next_state
            episode_reward += reward

            if total_steps > START_TRAINING_STEPS:
                agent.train(replay_buffer, BATCH_SIZE)
        
        # Use tqdm.write to print messages without breaking the progress bar.
        tqdm.write(f"Episode {episode+1} Complete. Reward: {episode_reward:.2f}")

        # Save a snapshot of the model periodically.
        if (episode + 1) % 50 == 0:
            tqdm.write(f"--- Saving model at episode {episode+1} ---")
            agent.save(f"models/ricc_hat_episode_{episode+1}")

    # Save the final trained model.
    agent.save("models/ricc_hat_final")
    print("\nTraining complete. Final model saved to 'models/ricc_hat_final'.")

# Corrected the syntax error in the last line.
if __name__ == '__main__':
    main()
