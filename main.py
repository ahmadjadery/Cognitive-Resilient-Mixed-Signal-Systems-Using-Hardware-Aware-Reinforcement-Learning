import os
import torch
import numpy as np
from tqdm import tqdm
from pll_env import PllEnv
from hat_trainer import TD3_HAT_Agent, ReplayBuffer

def main():
    # --- Hyperparameters ---
    MAX_EPISODES = 500  # You can change this to run longer or shorter
    START_TRAINING_STEPS = 5000
    EXPLORATION_NOISE = 0.1 # Reduced noise for the stable environment
    BATCH_SIZE = 256
    LR_ACTOR = 3e-4 # Learning rates can be adjusted
    LR_CRITIC = 3e-4
    SAVE_FREQ = 25

    # --- Setup ---
    CHECKPOINT_DIR = "models"
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    env = PllEnv()
    
    state_dim = env.reset().shape[0]
    action_dim = 2 # This should match your latest pll_env.py

    agent = TD3_HAT_Agent(state_dim=state_dim, action_dim=action_dim, 
                          lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, use_hat=True)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Logic to load and resume from a checkpoint
    start_episode = 0
    if os.path.exists(CHECKPOINT_FILE):
        print(f"--- Resuming training from checkpoint: {CHECKPOINT_FILE} ---")
        checkpoint = torch.load(CHECKPOINT_FILE)
        agent.load(CHECKPOINT_FILE, evaluate=False)
        start_episode = checkpoint.get('episode', 0)
        print(f"Resuming from episode {start_episode + 1}.")
    
    total_steps = agent.total_it
    for episode in tqdm(range(start_episode, MAX_EPISODES), initial=start_episode, total=MAX_EPISODES, desc="Total Episodes"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            total_steps += 1
            
            with torch.no_grad():
                action = (
                    agent.select_action(state)
                    + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
                ).clip(-1, 1)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, float(done))
            state = next_state
            episode_reward += reward

            if total_steps > START_TRAINING_STEPS:
                agent.train(replay_buffer, BATCH_SIZE)
        
        tqdm.write(f"Episode {episode+1} Complete. Reward: {episode_reward:.2f}")

        if (episode + 1) % SAVE_FREQ == 0:
            checkpoint_data = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_1_state_dict': agent.critic_1.state_dict(),
                'critic_2_state_dict': agent.critic_2.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'total_it': total_steps,
                'episode': episode + 1
            }
            torch.save(checkpoint_data, CHECKPOINT_FILE)
            tqdm.write(f"--- Checkpoint saved at episode {episode+1} ---")

    print("\nTraining complete.")

if __name__ == '__main__':
    main()