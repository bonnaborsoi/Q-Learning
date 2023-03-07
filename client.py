import numpy as np
import random
import threading
import time
import socket
from connection import connect
from connection import get_state_reward

# Definindo o número de estados e ações
num_estados = 96 # 24 x 4
num_acoes = 3

# Initializing the Q-table by all zeros
Q_table = np.zeros((num_estados, num_acoes)) 

# Define the hyperparameters
alpha = 0.8
gamma = 0.95
epsilon = 0.5

s = connect(2037) # Conecta-se ao servidor local do jogo

for i in range (1,100001):
    estado = 0b0000000 # Estado inicial
    done = False

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
        
        old_value = Q_table[estado, action]
        prox_estado, recompensa = get_state_reward(s , act)
        prox_estado = int(prox_estado, 2) # Converte para inteiro
        next_max = np.max(Q_table[prox_estado])

        new_value = (1 - alpha) * old_value + alpha * (recompensa + gamma * next_max)
        Q_table[estado, action] = new_value

        estado = prox_estado

        if estado == 0b0110111:
            done = True
        print(Q_table)
