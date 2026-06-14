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
    #Modelo
    m = gp.Model("NMPC")
    m.setParam("NonConvex", 2)

    #Variáveis
    x = m.addVars(T, nxu, lb=-16, ub=16, name="pos")
    x_bar = m.addVars(T, nxu, lb=-20, ub=20, name="x_bar")
    u = m.addVars(T, nxu, lb=-1, ub=1, name="u")
    lam = m.addVars(T, nlam, lb=0, name='Lambda')
    mu = m.addVars(T, nxu, name='mu')

    #Função objetivo
    obj = gp.quicksum(P[i,i] * x_bar[k,i]**2 for k in range(T) for i in range(nxu)) + gp.quicksum(Q[i,i] * u[k, i]**2 for k in range(T) for i in range(nxu))
    m.setObjective(obj, sense=gp.GRB.MINIMIZE)

    #=========================================================
    #Restrições
    #=========================================================

    #Pontos inicial e terminal
    for i in range(nxu):
        m.addConstr(x[0,i] == x_init[i] + u[0,i])
        # m.addConstr(x[T-1, i] == x_goal[i])

    #x_bark = xk - x_goal
    for k in range(T):
        for i in range(nxu):
            m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])

    #xk+1 = xk + uk
    for k in range(1, T):
        for i in range(nxu):
            m.addConstr(x[k, i] == x[k-1,i] + u[k, i])

    #Lambdak * b - mukk * xk >= 0
        m.addConstr(-gp.quicksum(lam[k, l] * b[l] for l in range(nlam) ) + gp.quicksum(mu[k, i] * x[k, i]  for i in range(nxu)) >= 1)

    # Lambdak * A - mukk  = 0
        for i in range(nxu):
            m.addConstr(gp.quicksum(lam[k, l] * A[l, i] for l in range(nlam)) - mu[k, i] == 0)

    #norma de mu
        m.addQConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nxu)) <= 1)

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

        return x_traj, u_traj

        fig, ax = plt.subplots(figsize=(8, 8))

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

        plt.show()


    else:
        print("Status:", m.status)
        return None, None




#Parametros
T = 10
nxu = 2
nlam = 4
num_rept = 5
k_pred = 5



#Matrizes multiplicadoras
P = np.eye(nxu)
Q = np.eye(nxu)

#Obstáculo
A = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
])
b = np.array([1,1,1,1])

# Lista para armazenamento das trajetórias
dataset_completo = []

#Pontos de interesse
# x_init = [-5, 0]
x_goal = [5, 0]

for n in range(num_rept):   
    while True:
        x1 = random.uniform(x_goal[0] - T, x_goal[0] + T)
        x2 = random.uniform(x_goal[1] - T, x_goal[1] + T)
        x_test = np.array([x1, x2])
        if not checa(A, b, x_test):
            x_init = x_test
            print(f'Funcionou {x1}, {x2}')   
            break

    estados, acoes = MPC(x_init)
    dados_da_sim = np.hstack((estados[:-1], acoes))
    dataset_completo.append(dados_da_sim)
    

dataset_completo = np.vstack(dataset_completo)
ds = pd.DataFrame(dataset_completo, columns=['x1', 'x2', 'u1', 'u2'])
ds.to_csv("dataset.csv", index= False)

