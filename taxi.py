import numpy as np
import random
import gymnasium as gym

env = gym.make("Taxi-v3", render_mode='ansi')

learning_rate = 0.1
gamma = 0.9
epsilon = 0.1
n_episodes = 10000

q_table = np.zeros([env.observation_space.n, env.action_space.n])

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state


def evaluate_agent(env, q_table, n_episodes=10):
    total_rewards = []
    total_steps = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        rewards = 0
        steps = 0

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)
            rewards += reward
            steps += 1

        total_rewards.append(rewards)
        total_steps.append(steps)

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    return avg_reward, avg_steps


avg_reward, avg_steps = evaluate_agent(env, q_table)
print(f"Average Reward: {avg_reward}, Average Steps: {avg_steps}")
