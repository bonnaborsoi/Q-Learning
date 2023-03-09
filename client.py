import numpy as np
import random
from connection import connect
from connection import get_state_reward
import os

# Initializing the Q-table (96 states x 3 actions)
# 96 states: 24 platforms x 4 directions. 3 actions: left, right and jump
def load_matrix():
    matrix = []
    with open(f'{os.getcwd()}/resultado.txt') as file:
        lines = file.readlines()
        for (idx, line) in enumerate(lines):
            values = line.replace('\n', '').split(' ')   
            values = [float(n) for n in values]
            matrix.append(values)
    return matrix

# Save matrix
def save_matrix(Q_table):
    with open(f'{os.getcwd()}/resultado.txt', 'w') as file:
        txt = ''
        for state in Q_table:
            txt += f'{round(state[0], 6)} {round(state[1], 6)} {round(state[2], 6)}\n'
        file.write(txt)

# Construct the Q-table 
Q_table = load_matrix()

# Define the hyperparameters
alpha = 0.6
gamma = 0.9
epsilon = 0.4
min_epsilon = 0.2
epsilon_decay = 0.005

victories = 0

reward_per_episode = []

# Connects to the game's local server
s = connect(2037) 

# Number of episodes = 100000
for episode in range (1,100001):
    state = 0b0000000 # Initial state
    state = int(state)
    done = False
    total_reward = 0

    while not done:
        
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0,2) # Select a random action
        else:
            action = np.argmax(Q_table[state]) # Exploit learned values
        if action == 0:
            act = "left"
        elif action == 1:
            act = "right"
        elif action == 2:
            act = "jump"

        old_value = Q_table[state][action]
        # Get the next state and reward
        next_state, reward = get_state_reward(s , act)

        next_state = int(next_state, 2) # Convert to integer
        next_max = np.max(Q_table[next_state])

        # Bellman Optimality Equation
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_table[state][action] = new_value
        total_reward = total_reward + reward

        # Update actual state
        state = next_state
        save_matrix(Q_table)

        # It's done if the agent falls or reachs the goal
        if reward == -100 or reward == 300:
            done = True
            break

    # Update the decay
    # epsilon = max(min_epsilon, epsilon*np.exp(-epsilon_decay*episode))
    epsilon -= epsilon_decay
    # Total reward when the agents reachs the goal and count number of victories
    if reward == 300:
        total_reward = 700 - total_reward
        victories = victories + 1

    # if epsilon*np.exp(-epsilon_decay*episode) <= min_epsilon:
    #     epsilon = 0.5

    if epsilon <= min_epsilon:
        epsilon = 0.5

    # Stores the total reward every time the agent falls or reachs the goal 
    reward_per_episode.append(total_reward)
