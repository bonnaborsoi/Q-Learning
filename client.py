import numpy as np
import random
import threading
import time
import socket
from connection import connect
from connection import get_state_reward
import os

def load_matrix():
    matrix = []
    with open(f'{os.getcwd()}/resultado.txt') as file:
        lines = file.readlines()
        for (idx, line) in enumerate(lines):
            values = line.replace('\n', '').split(' ')   
            values = [float(n) for n in values]
            matrix.append(values)
    return matrix

def save_matrix(Q_table):
    with open(f'{os.getcwd()}/resultado.txt', 'w') as file:
        txt = ''
        for estado in Q_table:
            txt += f'{round(estado[0], 6)} {round(estado[1], 6)} {round(estado[2], 6)}\n'
        file.write(txt)

# Definindo o número de estados e ações
num_estados = 96 # 24 x 4
num_acoes = 3

# Initializing the Q-table by all zeros
#Q_table = np.zeros((num_estados, num_acoes)) 
Q_table = load_matrix()

# Define the hyperparameters
alpha = 0.6
gamma = 0.9
epsilon = 1
decaimento_epsilon = 0.001
min_epsilon = 0.01


recompensa_por_episodio = []

s = connect(2037) # Conecta-se ao servidor local do jogo

for i in range (1,100001):
    estado = 0b0000000 # Estado inicial
    estado = int(estado)
    done = False
    recompensa_total = 0

    while not done:
        
        if random.uniform(0, 1) < epsilon or i == 1:
            action = random.randint(0,2) # Seleciona uma ação aleatória
        else:
            action = np.argmax(Q_table[estado]) # Exploit learned values
        if action == 0:
            act = "left"
        elif action == 1:
            act = "right"
        elif action == 2:
            act = "jump"
        
        old_value = Q_table[estado][action]
        prox_estado, recompensa = get_state_reward(s , act)
        prox_estado = int(prox_estado, 2) # Converte para inteiro
        next_max = np.max(Q_table[prox_estado])

        new_value = (1 - alpha) * old_value + alpha * (recompensa + gamma * next_max)
        Q_table[estado][action] = new_value
        recompensa_total = recompensa_total + recompensa
        #print(total_episode_reward)
        # print(f'Estado atual: {estado}')
        # print(f'Prox estado: {prox_estado}')
        # print(f'Recompensa: {recompensa}')

        estado = prox_estado
        save_matrix(Q_table)

        if recompensa == -100 or recompensa == 300:
            done = True
            break

    epsilon = max(min_epsilon, np.exp(-decaimento_epsilon*i))
    if recompensa == 300:
        recompensa_total = 1000 - recompensa_total
    recompensa_por_episodio.append(recompensa_total)
    # print(f'\n')
    print(f'epsilon: {epsilon}')
    print(f'recompensa do episodio: {recompensa_total}')
    print(f'{recompensa_por_episodio}')
    #print(rewards_per_episode)
