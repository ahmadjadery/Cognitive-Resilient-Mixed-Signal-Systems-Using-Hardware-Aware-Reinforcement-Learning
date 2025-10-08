import os
import torch
import numpy as np
from tqdm import tqdm
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent, ReplayBuffer

def main():
    # --- Hyperparameters ---
    MAX_EPISODES = 500
    START_TRAINING = 1000
    EXPLORATION_NOISE = 0.1
    BATCH_SIZE = 256
    
    # --- Setup ---
    env = PllEnv()
    state_dim = env.reset().shape[0]
    action_dim = 3
    
    agent = TD3_HAT_Agent(state_dim, action_dim, use_hat=True)
    # Update ReplayBuffer initialization
    replay_buffer = ReplayBuffer(state_dim, action_dim) 
    
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
            
            action = (
                agent.select_action(state)
                + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
            ).clip(-1, 1)

            next_state, reward, done, _ = env.step(action)
            
            # Update how data is added to the buffer
            replay_buffer.add(state, action, next_state, reward, float(done))
            
            state = next_state
            episode_reward += reward

            if total_steps > START_TRAINING:
                agent.train(replay_buffer, BATCH_SIZE)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode+1}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}")

    agent.save("models/ricc_hat_final")
    print("Training complete. Model saved.")

if __name__ == '__main__':
    main() # Corrected: removed extra parenthesis
