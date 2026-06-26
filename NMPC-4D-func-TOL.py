# %%
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import pandas as pd

def checa(A, b, x_pos):
    # Checa apenas as posições (X, Y) contra o obstáculo
    return np.all(A @ x_pos <= b)

def MPC(x_init):
    x_traj_real = np.zeros((T + 1, nx))
    u_traj_real = np.zeros((T, nu))
    x_traj_real[0, :] = x_init

    passos_executados = T

    for t in range(T):
        x_atual = x_traj_real[t, :]
        
        # Critério de parada do primeiro código: posição e velocidade próximas de zero no alvo
        if np.linalg.norm(x_atual[0:2] - x_goal[0:2]) < 0.25 and np.linalg.norm(x_atual[2:4]) < 0.40:
            print(f"\n[SUCESSO] Alvo alcançado de forma estável no passo t={t}!")
            passos_executados = t
            # Ajusta o tamanho das matrizes para o que foi de fato executado
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

        # Função objetivo considerando estados (posição e velocidade) e esforço de controle
        obj = gp.quicksum(P * x_bar[k, i]**2 for k in range(1, k_pred + 1) for i in range(nx)) + \
              gp.quicksum(Q * u[k, i]**2 for k in range(k_pred) for i in range(nu))
        m.setObjective(obj, sense=gp.GRB.MINIMIZE)

        for i in range(nx):
            m.addConstr(x[0, i] == x_atual[i])

        for k in range(k_pred):
            # Integração da Dinâmica 4D (Duplo Integrador com DT)
            # Posição_futura = Posição_atual + Velocidade_atual * DT
            m.addConstr(x[k + 1, 0] == x[k, 0] + x[k, 2] * dt)
            m.addConstr(x[k + 1, 1] == x[k, 1] + x[k, 3] * dt)
            # Velocidade_futura = Velocidade_atual + Aceleração(u) * DT
            m.addConstr(x[k + 1, 2] == x[k, 2] + u[k, 0] * dt)
            m.addConstr(x[k + 1, 3] == x[k, 3] + u[k, 1] * dt)
            
            # Restrições KKT para evitar o obstáculo (baseado apenas na posição x[k, 0:2])
            m.addConstr(gp.quicksum(-lam[k, j] * b[j] for j in range(nlam)) + 
                        mu[k, 0] * x[k, 0] + mu[k, 1] * x[k, 1] >= 1)

            for i in range(nu):
                m.addConstr(gp.quicksum(lam[k, j] * A[j, i] for j in range(nlam)) - mu[k, i] == 0)

            m.addQConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nu)) <= 1)

        for k in range(k_pred + 1):
            for i in range(nx):
                m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])

        # Restrição terminal do primeiro código (suavizar a chegada)
        distancia_atual = np.linalg.norm(x_atual[0:2] - x_goal[0:2])
        if distancia_atual < 4:
            m.addQConstr(x_bar[k_pred, 0]**2 + x_bar[k_pred, 1]**2 <= 0.25**2)
            m.addQConstr(x_bar[k_pred, 2]**2 + x_bar[k_pred, 3]**2 <= 0.15**2)

        m.optimize()

        if m.status == gp.GRB.OPTIMAL:
            u_aplicar = np.array([u[0, 0].X, u[0, 1].X])
            u_traj_real[t, :] = u_aplicar
            
            # Evolução do sistema real com a dinâmica 4D
            x_traj_real[t + 1, 0] = x_traj_real[t, 0] + x_traj_real[t, 2] * dt
            x_traj_real[t + 1, 1] = x_traj_real[t, 1] + x_traj_real[t, 3] * dt
            x_traj_real[t + 1, 2] = x_traj_real[t, 2] + u_aplicar[0] * dt
            x_traj_real[t + 1, 3] = x_traj_real[t, 3] + u_aplicar[1] * dt
        else:
            print(f"Simulação falhou no passo {t} com status: {m.status}")
            return None, None, t

    return x_traj_real, u_traj_real, passos_executados

# Configurações básicas adaptadas
T = 20       # Horizonte total maior para a dinâmica 4D amortecer
dt = 0.5        # Passo de tempo inserido
nx = 4          # Estado agora é 4D (Pos_X, Pos_Y, Vel_X, Vel_Y)
nu = 2          # Controle é 2D (Accel_X, Accel_Y)
nlam = 4        
num_rept = 100   
k_pred = 12     # Horizonte de predição do NMPC estendido

P = 50
Q = 1

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.array([1, 1, 1, 1])

# Alvo 4D (Posição em [5,0] e velocidades desejadas em 0)
x_goal = np.array([5.0, 0.0, 0.0, 0.0])
dataset_completo = []
ultima_trajetoria_completa = None

for n in range(num_rept):   
    while True:
        # Sorteia apenas posições iniciais fora do obstáculo
        x1 = random.uniform(-6, -4)
        x2 = random.uniform(-1, 1)
        if not checa(A, b, np.array([x1, x2])):
            # Inicia com posições sorteadas e velocidades zeradas [x1, x2, 0, 0]
            x_init = np.array([x1, x2, 0.0, 0.0])
            print(f'Ponto inicial: [{x1:.2f}, {x2:.2f}]')   
            break

    estados, acoes, passos = MPC(x_init)
    
    if estados is not None and acoes is not None:
        ultima_trajetoria_completa = estados  
        dados_da_sim = np.hstack((estados[:-1], acoes))
        dataset_completo.append(dados_da_sim)

# Salva o arquivo CSV adaptado para 4D + 2D
if len(dataset_completo) > 0:
    dataset_completo = np.vstack(dataset_completo)
    ds = pd.DataFrame(dataset_completo, columns=['x1', 'x2', 'v1', 'v2', 'u1', 'u2'])
    ds.to_csv("dataset.csv", index=False)
    print("Dataset salvo com sucesso.")

# Bloco de Plotagem adaptado para ler estados 4D
if num_rept == 1 and ultima_trajetoria_completa is not None:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Posições X e Y são os índices 0 e 1 de ultima_trajetoria_completa
    ax.plot(ultima_trajetoria_completa[:, 0], ultima_trajetoria_completa[:, 1], 'bo-', linewidth=2, label='Trajetória Real (NMPC 4D)')
    ax.plot(ultima_trajetoria_completa[0, 0], ultima_trajetoria_completa[0, 1], 'gs', markersize=10, label='Inicial')
    ax.plot(x_goal[0], x_goal[1], 'rs', markersize=10, label='Objetivo')

    # Desenha o obstáculo central
    square = patches.Rectangle((-1, -1), 2, 2, edgecolor='black', facecolor='red', alpha=0.4, label='Obstáculo')
    ax.add_patch(square)

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_title('Validação Visual do NMPC Unificado (Dinâmica 4D)')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.show()