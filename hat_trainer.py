import torch
import numpy as np
from ricc_arch import Actor, Critic # Assuming Critic is defined in ricc_arch.py

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in ind:
            s, a, s_prime, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_prime, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))
        return (np.array(states), np.array(actions), np.array(next_states), 
                np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1))

class TD3_HAT_Agent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, discount=0.99, tau=0.005, use_hat=True):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # In a real TD3, there would be two critics
        self.critic = Critic(state_dim, action_dim) 
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.discount = discount
        self.tau = tau
        self.use_hat = use_hat
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        # Pass use_hat flag to the forward method
        action = self.actor(state, use_hat=self.use_hat).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        
        # --- Update Critic ---
        # Clipped Double-Q Learning target
        with torch.no_grad():
            next_action = self.actor_target(next_state, use_hat=self.use_hat)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.discount * target_Q
            
        current_Q = self.critic(state, action)
        critic_loss = torch.nn.functional.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Delayed Actor Update ---
        actor_loss = -self.critic(state, self.actor(state, use_hat=self.use_hat)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Target Networks ---
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
