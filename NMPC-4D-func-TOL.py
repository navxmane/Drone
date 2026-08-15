# %%
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import pandas as pd

def checa(A, b, x_pos):
    return np.all(A @ x_pos <= b)

def MPC(x_init):
    x_traj_real = np.zeros((T + 1, nx))
    u_traj_real = np.zeros((T, nu))
    x_traj_real[0, :] = x_init

    passos_executados = T

    for t in range(T):
        x_atual = x_traj_real[t, :]
        
        # Critério de parada
        if np.linalg.norm(x_atual[0:2] - x_goal[0:2]) < 0.25 and np.linalg.norm(x_atual[2:4]) < 0.40:
            print(f"\n[SUCESSO] Alvo alcançado de forma estável no passo t={t}!")
            passos_executados = t
            x_traj_real = x_traj_real[:passos_executados + 1, :]
            u_traj_real = u_traj_real[:passos_executados, :]
            break

        m = gp.Model(f"NMPC_Step_{t}")
        m.setParam("OutputFlag", 0) 
        m.setParam("NonConvex", 2)

        # Variáveis de decisão (Estados 4D e Controle 2D)
        x = m.addVars(k_pred + 1, nx, lb=-10, ub=10, name="pos")
        x_bar = m.addVars(k_pred + 1, nx, lb=-20, ub=20, name="x_bar")
        u = m.addVars(k_pred, nu, lb=-1.5, ub=1.5, name="u")
        
        # Multiplicadores de Lagrange
        lam = m.addVars(k_pred, nlam, lb=0, name='Lambda')
        mu = m.addVars(k_pred, nu, lb=-gp.GRB.INFINITY, name='mu')

        # 1. CRIAR VARIÁVEIS DE FOLGA (SLACK VARIABLES)
        s = m.addVars(k_pred, lb=0.0, ub=gp.GRB.INFINITY, name="slack")

        # PESO DA PENALIZAÇÃO (Ex: 1e4 ou 1e5 para priorizar fortemente a evasão)
        W_slack = 1e5

        # 2. FUNÇÃO OBJETIVO COM PENALIZAÇÃO QUADRÁTICA DA FOLGA
        P_term = 3.0 * P
        obj = (gp.quicksum(P * x_bar[k, i]**2 for k in range(1, k_pred) for i in range(nx)) +
               gp.quicksum(P_term * x_bar[k_pred, i]**2 for i in range(nx)) +
               gp.quicksum(Q * u[k, i]**2 for k in range(k_pred) for i in range(nu)) +
               gp.quicksum(W_slack * s[k]**2 for k in range(k_pred))) # Penalização do Slack
        
        m.setObjective(obj, sense=gp.GRB.MINIMIZE)

        for i in range(nx):
            m.addConstr(x[0, i] == x_atual[i])

        for k in range(k_pred):
            # Integração da Dinâmica 4D
            m.addConstr(x[k + 1, 0] == x[k, 0] + x[k, 2] * dt)
            m.addConstr(x[k + 1, 1] == x[k, 1] + x[k, 3] * dt)
            m.addConstr(x[k + 1, 2] == x[k, 2] + u[k, 0] * dt)
            m.addConstr(x[k + 1, 3] == x[k, 3] + u[k, 1] * dt)
            
            # 3. RESTRIÇÃO KKT SUAVIZADA COM SLACK
            m.addConstr(gp.quicksum(-lam[k, j] * b[j] for j in range(nlam)) + 
                        mu[k, 0] * x[k, 0] + mu[k, 1] * x[k, 1] + s[k] >= 1)

            for i in range(nu):
                m.addConstr(gp.quicksum(lam[k, j] * A[j, i] for j in range(nlam)) - mu[k, i] == 0)

            m.addQConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nu)) <= 1)

        for k in range(k_pred + 1):
            for i in range(nx):
                m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])

        # Restrição terminal suave (também pode receber slack se necessário)
        distancia_atual = np.linalg.norm(x_atual[0:2] - x_goal[0:2])
        if distancia_atual < 2.0:
            m.addQConstr(x_bar[k_pred, 0]**2 + x_bar[k_pred, 1]**2 <= 0.35**2)
            m.addQConstr(x_bar[k_pred, 2]**2 + x_bar[k_pred, 3]**2 <= 0.35**2)

        m.optimize()

        if m.status == gp.GRB.OPTIMAL:
            u_aplicar = np.array([u[0, 0].X, u[0, 1].X])
            u_traj_real[t, :] = u_aplicar
            
            # Checa se o slack foi ativado no primeiro passo
            if s[0].X > 1e-4:
                print(f"[Aviso t={t}] Restrição de obstáculo suavizada! Slack s[0] = {s[0].X:.4f}")

            # Evolução do sistema real (com inclusão de perturbação estocástica para teste de robustez)
            ruido_vento = np.random.normal(0, 0.02, size=2) # Exemplo de perturbação externa
            x_traj_real[t + 1, 0] = x_traj_real[t, 0] + x_traj_real[t, 2] * dt
            x_traj_real[t + 1, 1] = x_traj_real[t, 1] + x_traj_real[t, 3] * dt
            x_traj_real[t + 1, 2] = x_traj_real[t, 2] + u_aplicar[0] * dt + ruido_vento[0]
            x_traj_real[t + 1, 3] = x_traj_real[t, 3] + u_aplicar[1] * dt + ruido_vento[1]
        else:
            print(f"Simulação falhou no passo {t} com status: {m.status}")
            return None, None, t

    return x_traj_real, u_traj_real, passos_executados

# Parâmetros da Simulação
T = 25          
dt = 0.5        
nx = 4          
nu = 2          
nlam = 4        
num_rept = 1  
k_pred = 7     

P = 0.5
Q = 1.0

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.array([1, 1, 1, 1])

x_goal = np.array([5.0, 0.0, 0.0, 0.0])

# Teste com posição inicial muito crítica
x_init = np.array([-2.0, 0.0, 0.0, 0.0])
print(f'Ponto inicial: [{x_init[0]:.2f}, {x_init[1]:.2f}]')   

estados, acoes, passos = MPC(x_init)

# Plotagem
if estados is not None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(estados[:, 0], estados[:, 1], 'bo-', linewidth=2, label='Trajetória NMPC (com Slack)')
    ax.plot(estados[0, 0], estados[0, 1], 'gs', markersize=10, label='Inicial')

    square = patches.Rectangle((-1, -1), 2, 2, edgecolor='black', facecolor='red', alpha=0.7, label='Obstáculo')
    ax.add_patch(square)

    square_goal = patches.Rectangle((4.75, -0.25), 0.5, 0.5, edgecolor='black', facecolor='yellow', label='Estado objetivo')
    ax.add_patch(square_goal)
    
    plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_title('NMPC 4D com Restrições Suaves (Slack Variables)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.show()