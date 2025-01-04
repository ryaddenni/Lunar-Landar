import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer
from q_network import QNetwork

# Hyperparameters and configurations
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration factor (epsilon-greedy)
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
replay_buffer_capacity = 100000
epochs = 1000
max_timesteps = 500

input_size = 8  # LunarLander-v2 state space size
output_size = 4  # LunarLander-v2 action space size

def train_dqn():
    global epsilon
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    q_network = QNetwork(input_size, output_size)
    target_q_network = QNetwork(input_size, output_size)
    target_q_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    for epoch in range(epochs):
        if epoch % 50 == 0:
            env = gym.make("LunarLander-v2", render_mode="human")
        else:
            env = gym.make("LunarLander-v2")

        try:
            state, _ = env.reset()
            state = np.array(state).flatten()
            done = False
            timestep = 0

            while not done and timestep < max_timesteps:
                timestep += 1
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action = torch.argmax(q_network(state_tensor)).item()

                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = np.array(next_state).flatten()
                replay_buffer.push(state, action, reward, next_state, terminated or truncated)

                if replay_buffer.size() >= batch_size:
                    transitions = replay_buffer.sample(batch_size)
                    batch = list(zip(*transitions))
                    states, actions, rewards, next_states, dones = batch

                    states_tensor = torch.FloatTensor(np.array(states))
                    actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                    rewards_tensor = torch.FloatTensor(rewards)
                    next_states_tensor = torch.FloatTensor(np.array(next_states))
                    dones_tensor = torch.BoolTensor(dones)

                    current_q_values = q_network(states_tensor).gather(1, actions_tensor)
                    next_q_values = target_q_network(next_states_tensor).max(1)[0].detach()
                    target_q_values = rewards_tensor + (gamma * next_q_values * (~dones_tensor))

                    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                state = next_state
                if timestep % 10 == 0:
                    target_q_network.load_state_dict(q_network.state_dict())
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                if terminated or truncated:
                    break

            print(f"Epoch {epoch+1}/{epochs}, Epsilon: {epsilon:.2f}, Timestep: {timestep}")

        finally:
            # Explicitly close the current environment
            env.close()

if __name__ == "__main__":
    train_dqn()
