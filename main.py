import os
import torch
import numpy as np
from tqdm import tqdm
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent, ReplayBuffer

def main():
    # --- Hyperparameters ---
    MAX_EPISODES = 500
    START_TRAINING = 1000  # Number of steps before training starts
    EXPLORATION_NOISE = 0.1
    BATCH_SIZE = 256
    
    # --- Setup ---
    env = PllEnv()
    state_dim = env.reset().shape[0]
    action_dim = 3 # delta_icp, delta_kp, delta_ki
    
    agent = TD3_HAT_Agent(state_dim, action_dim, use_hat=True)
    replay_buffer = ReplayBuffer()
    
    if not os.path.exists("models"):
        os.makedirs("models")

    # --- Training Loop ---
    total_steps = 0
    for episode in tqdm(range(MAX_EPISODES)):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            total_steps += 1
            
            # Select action with exploration noise
            action = agent.select_action(state)
            noise = np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
            action = (action + noise).clip(-1, 1)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, next_state, reward, float(done)))
            
            state = next_state
            episode_reward += reward

            # Train agent
            if total_steps > START_TRAINING:
                agent.train(replay_buffer, BATCH_SIZE)
        
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}")

    # Save the final trained model
    agent.save("models/ricc_hat_final")

if __name__ == '__main__':
    main()
