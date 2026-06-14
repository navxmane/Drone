# %%
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import pandas as pd

def checa(A, b, x):
    return np.all(A @ x <= b)

def MPC(x_init):
    x_real[0,:] = x_init

    for t in  range(T):
        x_atual = x_real[t, :]
    #Modelo
        m = gp.Model("NMPC")
        m.setParam("NonConvex", 2)

        #Variáveis
        x = m.addVars(k_pred, nxu, lb=-16, ub=16, name="pos")
        x_bar = m.addVars(k_pred, nxu, lb=-20, ub=20, name="x_bar")
        u = m.addVars(k_pred, nxu, lb=-1, ub=1, name="u")
        lam = m.addVars(k_pred, nlam, lb=0, name='Lambda')
        mu = m.addVars(k_pred, nxu, name='mu')

        #Função objetivo
        obj = gp.quicksum(P * x_bar[k,i]**2 for k in range(k_pred) for i in range(nxu)) + gp.quicksum(Q * u[k, i]**2 for k in range(k_pred) for i in range(nxu))
        m.setObjective(obj, sense=gp.GRB.MINIMIZE)

        #=========================================================
        #Restrições
        #=========================================================

        #Pontos inicial e terminal
        for i in range(nxu):
            m.addConstr(x[0,i] == x_atual[i] + u[0,i])
            # m.addConstr(x[T-1, i] == x_goal[i])

        for k in range(1,k_pred):
        #x_bark = xk - x_goal
            m.addConstr(x_bar[k] == x[k] - x_goal)

            #xk+1 = xk + uk
            m.addConstr(x[k] == x[k-1] + u[k])

            #Lambdak * b - mukk * xk >= 0
            m.addConstr(-lam[k] @ b + mu[k] @ x[k] >= 1)

            # Lambdak * A - mukk  = 0
            m.addConstr(A.T @ lam[k] - mu[k] == 0)

            #norma de mu
            m.addQConstr(mu[k] @ mu[k] <= 1)
        
        m.addConstr(x_bar[k_pred] == x[k_pred] - x_goal)

        m.optimize()

        #====================================================================
        #Plotagem se ótimo é encontrado
        #====================================================================
        if m.status == gp.GRB.OPTIMAL:
            print("Ótimo encontrado")
            # Criamos a lista completa com todos os pontos
            todos_pontos = [x_init] + [[x[k, 0].X, x[k, 1].X] for k in range(T)]

            # Imprimimos formatado de P0 até P10
            for i, ponto in enumerate(todos_pontos):
                print(f"P{i} = [{ponto[0]:.2f}, {ponto[1]:.2f}]")

            x_traj = np.array(todos_pontos)
            u_traj = [[u[k, 0].X, u[k,1].X] for k in range(T)]

            u_aplicar = np.array([u[0,0].X, u[0,1].X])

            u_real[t,:] = u_aplicar

            x_real[t+1,:] = x_real[t,:] + u_aplicar

        else:
            print("Status:", m.status)
            return None, None

    return x_traj, u_traj

    ''' fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        x_traj[:, 0],
        x_traj[:, 1],
        'bo-',
        linewidth=2,
        label='Trajetória'
    )

    ax.plot(
        x_init[0],
        x_init[1],
        'gs',
        markersize=10,
        label='Inicial'
    )

    ax.plot(
        x_goal[0],
        x_goal[1],
        'rs',
        markersize=10,
        label='Objetivo'
    )

    square = patches.Rectangle((-1, -1), 2, 2, edgecolor='black', facecolor='red')
    ax.add_patch(square)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_title('Trajetória ótima')

    ax.grid(True)
    ax.axis('equal')
    ax.legend()

    plt.show()'''







#Parametros
T = 10
nxu = 2
nlam = 4
num_rept = 1
k_pred = 5



#Matrizes multiplicadoras
P = 50
Q = 1

#Obstáculo
A = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
])
b = np.array([1,1,1,1])

# Declaração do x atual (real)
x_real = np.zeros((T, nxu))
u_real = np.zeros((T, nxu))

# Lista para armazenamento das trajetórias
dataset_completo = []

#Pontos de interesse
x_init = [-5, 0]
x_goal = [5, 0]

for n in range(num_rept):   
    while True:
        x1 = random.uniform(x_goal[0] - T, x_goal[0] + T)
        x2 = random.uniform(x_goal[1] - T, x_goal[1] + T)
        x_test = np.array([x1, x2])
        if not checa(A, b, x_test):
            # x_init = x_test
            print(f'Funcionou {x1}, {x2}')   
            break

    x_real[0,:] = x_init
    estados, acoes = MPC(x_init)
    dados_da_sim = np.hstack((estados[:-1], acoes))
    dataset_completo.append(dados_da_sim)
    

dataset_completo = np.vstack(dataset_completo)
ds = pd.DataFrame(dataset_completo, columns=['x1', 'x2', 'u1', 'u2'])
ds.to_csv("dataset.csv", index= False)

