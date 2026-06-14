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
    x_traj_real = np.zeros((T + 1, nxu))
    u_traj_real = np.zeros((T, nxu))
    x_traj_real[0, :] = x_init

    for t in range(T):
        x_atual = x_traj_real[t, :]
        
        m = gp.Model(f"NMPC_Step_{t}")
        m.setParam("OutputFlag", 0) 
        m.setParam("NonConvex", 2)

        x = m.addVars(k_pred + 1, nxu, lb=-16, ub=16, name="pos")
        x_bar = m.addVars(k_pred + 1, nxu, lb=-20, ub=20, name="x_bar")
        u = m.addVars(k_pred, nxu, lb=-1, ub=1, name="u")
        lam = m.addVars(k_pred, nlam, lb=0, name='Lambda')
        mu = m.addVars(k_pred, nxu, lb=-gp.GRB.INFINITY, name='mu')

        obj = gp.quicksum(P * x_bar[k, i]**2 for k in range(1, k_pred + 1) for i in range(nxu)) + \
              gp.quicksum(Q * u[k, i]**2 for k in range(k_pred) for i in range(nxu))
        m.setObjective(obj, sense=gp.GRB.MINIMIZE)

        for i in range(nxu):
            m.addConstr(x[0, i] == x_atual[i])

        for k in range(k_pred):
            for i in range(nxu):
                m.addConstr(x[k + 1, i] == x[k, i] + u[k, i])
            
            m.addConstr(gp.quicksum(-lam[k, j] * b[j] for j in range(nlam)) + 
                        gp.quicksum(mu[k, i] * x[k, i] for i in range(nxu)) >= 1)

            for i in range(nxu):
                m.addConstr(gp.quicksum(lam[k, j] * A[j, i] for j in range(nlam)) - mu[k, i] == 0)

            m.addQConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nxu)) <= 1)

        for k in range(k_pred + 1):
            for i in range(nxu):
                m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])

        m.optimize()

        if m.status == gp.GRB.OPTIMAL:
            u_aplicar = np.array([u[0, 0].X, u[0, 1].X])
            u_traj_real[t, :] = u_aplicar
            x_traj_real[t + 1, :] = x_traj_real[t, :] + u_aplicar
        else:
            print(f"Simulação falhou no passo {t} com status: {m.status}")
            return None, None

    return x_traj_real, u_traj_real

# Configurações básicas
T = 15          # Aumentado para 15 passos para dar tempo de desviar e chegar
nxu = 2         
nlam = 4        
num_rept = 1    # Defina como 1 para testar um único ponto e gerar o gráfico
k_pred = 5      

P = 50
Q = 1

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.array([1, 1, 1, 1])

x_goal = [5, 0]
dataset_completo = []
ultima_trajetoria_completa = None  # Variável para guardar os pontos do gráfico

for n in range(num_rept):   
    while True:
        # Sorteia pontos iniciais
        x1 = random.uniform(x_goal[0] - T, x_goal[0] + T)
        x2 = random.uniform(x_goal[1] - T, x_goal[1] + T)
        x_test = np.array([x1, x2])
        if not checa(A, b, x_test):
            x_init = x_test
            print(f'Ponto inicial: [{x1:.2f}, {x2:.2f}]')   
            break

    estados, acoes = MPC(x_init)
    
    if estados is not None and acoes is not None:
        ultima_trajetoria_completa = estados  # Guarda para o gráfico (contém o T+1 final)
        dados_da_sim = np.hstack((estados[:-1], acoes))
        dataset_completo.append(dados_da_sim)

# Salva o arquivo CSV normalmente
if len(dataset_completo) > 0:
    dataset_completo = np.vstack(dataset_completo)
    ds = pd.DataFrame(dataset_completo, columns=['x1', 'x2', 'u1', 'u2'])
    ds.to_csv("dataset.csv", index=False)
    print("Dataset salvo com sucesso.")

# ====================================================================
# Bloco condicional de Plotagem: Ativado apenas se rodar para 1 único ponto
# ====================================================================
if num_rept == 1 and ultima_trajetoria_completa is not None:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plota usando a trajetória completa salva antes do filtro do dataset
    ax.plot(ultima_trajetoria_completa[:, 0], ultima_trajetoria_completa[:, 1], 'bo-', linewidth=2, label='Trajetória Real (MPC)')
    ax.plot(ultima_trajetoria_completa[0, 0], ultima_trajetoria_completa[0, 1], 'gs', markersize=10, label='Inicial')
    ax.plot(x_goal[0], x_goal[1], 'rs', markersize=10, label='Objetivo')

    # Desenha o obstáculo central
    square = patches.Rectangle((-1, -1), 2, 2, edgecolor='black', facecolor='red', alpha=0.4, label='Obstáculo')
    ax.add_patch(square)

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_title('Validação Visual do NMPC (Horizonte Rolante)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.show()