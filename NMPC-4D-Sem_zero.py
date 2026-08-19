# %% 
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
import time

def checa(A, b, x_pos):
    return np.all(A @ x_pos <= b)

def MPC(x_init, k_pred):
    x_traj_real = np.zeros((T + 1, nx))
    u_traj_real = np.zeros((T, nu))
    x_traj_real[0, :] = x_init

    prev_x = None
    prev_x_bar = None
    prev_u = None
    prev_lam = None
    prev_mu = None

    passos_executados = T
    telemetria_passos = []

    for t in range(T):
        x_atual = x_traj_real[t, :]
        
        # Critério de parada ao atingir a meta
        if np.linalg.norm(x_atual[0:2] - x_goal[0:2]) < 0.25 and np.linalg.norm(x_atual[2:4]) < 0.40:
            passos_executados = t
            x_traj_real = x_traj_real[:passos_executados + 1, :]
            u_traj_real = u_traj_real[:passos_executados, :]
            break

        m = gp.Model(f"NMPC_Step_{t}")
        m.setParam("OutputFlag", 0) 
        m.setParam("NonConvex", 2)
      
        x = m.addVars(k_pred + 1, nx, lb=-10, ub=10, name="pos")
        x_bar = m.addVars(k_pred + 1, nx, lb=-20, ub=20, name="x_bar")
        u = m.addVars(k_pred, nu, lb=-1.5, ub=1.5, name="u")
        lam = m.addVars(k_pred, nlam, lb=0, name='Lambda')
        mu = m.addVars(k_pred, nu, lb=-gp.GRB.INFINITY, name='mu')

        obj = (gp.quicksum(P * x_bar[k, i]**2 for k in range(1, k_pred + 1) for i in range(nx)) +
            #    gp.quicksum(P_mat[i,j] * x_bar[k_pred, i] * x_bar[k_pred, j] for i in range(nx) for j in range(nx)) +
               gp.quicksum(Q * u[k, i]**2 for k in range(k_pred) for i in range(nu)))
        m.setObjective(obj, sense=gp.GRB.MINIMIZE)

        # Condição inicial do passo
        for i in range(nx):
            m.addConstr(x[0, i] == x_atual[i])

        # Dinâmica e restrições de obstáculos
        for k in range(k_pred):
            m.addConstr(x[k + 1, 0] == x[k, 0] + x[k, 2] * dt)
            m.addConstr(x[k + 1, 1] == x[k, 1] + x[k, 3] * dt)
            m.addConstr(x[k + 1, 2] == x[k, 2] + u[k, 0] * dt)
            m.addConstr(x[k + 1, 3] == x[k, 3] + u[k, 1] * dt)
            
            m.addConstr(gp.quicksum(-lam[k, j] * b[j] for j in range(nlam)) + 
                        mu[k, 0] * x[k, 0] + mu[k, 1] * x[k, 1] >= 1)

            for i in range(nu):
                m.addConstr(gp.quicksum(lam[k, j] * A[j, i] for j in range(nlam)) - mu[k, i] == 0)

            m.addQConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nu)) <= 1)

        # Variável de erro (tracking)
        for k in range(k_pred + 1):
            for i in range(nx):
                m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])

        # Restrição terminal quando próximo do objetivo
        distancia_atual = np.linalg.norm(x_atual[0:2] - x_goal[0:2])
        if distancia_atual < 1.5:
            m.addQConstr(x_bar[k_pred, 0]**2 + x_bar[k_pred, 1]**2 <= 0.25**2)
            m.addQConstr(x_bar[k_pred, 2]**2 + x_bar[k_pred, 3]**2 <= 0.15**2)

        # --- WARM-START ---
        if t > 0 and prev_u is not None:
            # Desloca a solução anterior em 1 passo no tempo
            for k in range(k_pred):
                for i in range(nx):
                    x[k, i].Start = prev_x[k + 1, i]
                    x_bar[k, i].Start = prev_x_bar[k + 1, i]

            # Extrapola o último estado no horizonte
            x_term_start = np.array([
                prev_x[k_pred, 0] + prev_x[k_pred, 2] * dt,
                prev_x[k_pred, 1] + prev_x[k_pred, 3] * dt,
                prev_x[k_pred, 2] + prev_u[k_pred - 1, 0] * dt,
                prev_x[k_pred, 3] + prev_u[k_pred - 1, 1] * dt
            ])
            for i in range(nx):
                x[k_pred, i].Start = x_term_start[i]
                x_bar[k_pred, i].Start = x_term_start[i] - x_goal[i]

            # Shift nos controles e duals (k = 0 até k_pred - 2)
            for k in range(k_pred - 1):
                for i in range(nu):
                    u[k, i].Start = prev_u[k + 1, i]
                    mu[k, i].Start = prev_mu[k + 1, i]
                for j in range(nlam):
                    lam[k, j].Start = prev_lam[k + 1, j]

            # Mantém o último controle repetido no final do horizonte
            for i in range(nu):
                u[k_pred - 1, i].Start = prev_u[k_pred - 1, i]
                mu[k_pred - 1, i].Start = prev_mu[k_pred - 1, i]
            for j in range(nlam):
                lam[k_pred - 1, j].Start = prev_lam[k_pred - 1, j]
                
        m.optimize()

        # Telemetria do Gurobi
        tempo_solver = m.Runtime
        nos_explorados = int(m.NodeCount)
        
        try:
            gap = m.MIPGap
        except AttributeError:
            gap = 0.0

        telemetria_passos.append({
            'Passo': t,
            'Tempo_Solver_s': tempo_solver,
            'Nos_Explorados': nos_explorados,
            'Gap': gap
        })

        if m.status == gp.GRB.OPTIMAL:
            u_aplicar = np.array([u[0, 0].X, u[0, 1].X])
            u_traj_real[t, :] = u_aplicar
            
            # Atualiza dinâmica do sistema
            x_traj_real[t + 1, 0] = x_traj_real[t, 0] + x_traj_real[t, 2] * dt
            x_traj_real[t + 1, 1] = x_traj_real[t, 1] + x_traj_real[t, 3] * dt
            x_traj_real[t + 1, 2] = x_traj_real[t, 2] + u_aplicar[0] * dt
            x_traj_real[t + 1, 3] = x_traj_real[t, 3] + u_aplicar[1] * dt

            # Salva variáveis para o próximo warm-start
            prev_x = np.array([[x[k, i].X for i in range(nx)] for k in range(k_pred + 1)])
            prev_x_bar = np.array([[x_bar[k, i].X for i in range(nx)] for k in range(k_pred + 1)])
            prev_u = np.array([[u[k, i].X for i in range(nu)] for k in range(k_pred)])
            prev_lam = np.array([[lam[k, j].X for j in range(nlam)] for k in range(k_pred)])
            prev_mu = np.array([[mu[k, i].X for i in range(nu)] for k in range(k_pred)])
        else:
            print(f"Falhou para k_pred={k_pred} no passo {t}")
            return None, None, t, pd.DataFrame(telemetria_passos)

    df_telemetria = pd.DataFrame(telemetria_passos)
    return x_traj_real, u_traj_real, passos_executados, df_telemetria

# --- Parâmetros ---
T = 25
dt = 0.5
nx = 4
nu = 2
nlam = 4

P = 10
Q = 10

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.array([1, 1, 1, 1])

x_init = np.array([-5.0, 0.0, 0.0, 0.0])
x_goal = np.array([5.0, 0.0, 0.0, 0.0])

# Execução da simulação
# Execução da simulação e exportação para CSV
lista_k_pred = [3,5,7,10,12]
trajetorias = {}

for kp in lista_k_pred:
    print(f"Simulando NMPC para k_pred = {kp}...")
    estados, acoes, passos, df_telemetria = MPC(x_init, k_pred=kp)
    
    if estados is not None:
        trajetorias[kp] = (estados, acoes)
        
        # --- EXPORTAÇÃO TELEMETRIA PARA CSV ---
        nome_arquivo = f"K_pred_{kp}_warm-start.csv"
        df_telemetria.to_csv(nome_arquivo, index=False)
        print(f"\nTelemetria salva em: {nome_arquivo}")
        
        # Impressão no terminal
        print("\n--- Resumo de Telemetria do Gurobi ---")
        print(df_telemetria.to_string(index=False))
        print(f"\nTempo Total Solver: {df_telemetria['Tempo_Solver_s'].sum():.4f} s")
        print(f"Total de Nós Explorados: {df_telemetria['Nos_Explorados'].sum()}")

# %%
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
import time


def checa(A, b, x_pos):
    return np.all(A @ x_pos <= b)

def MPC(x_init, k_pred):
    x_traj_real = np.zeros((T + 1, nx))
    u_traj_real = np.zeros((T, nu))
    x_traj_real[0, :] = x_init

    passos_executados = T
    telemetria_passos = []

    for t in range(T):
        x_atual = x_traj_real[t, :]
        
        if np.linalg.norm(x_atual[0:2] - x_goal[0:2]) < 0.25 and np.linalg.norm(x_atual[2:4]) < 0.40:
            passos_executados = t
            x_traj_real = x_traj_real[:passos_executados + 1, :]
            u_traj_real = u_traj_real[:passos_executados, :]
            break

        m = gp.Model(f"NMPC_Step_{t}")
        m.setParam("OutputFlag", 0) 
        m.setParam("NonConvex", 2)

        x = m.addVars(k_pred + 1, nx, lb=-10, ub=10, name="pos")
        x_bar = m.addVars(k_pred + 1, nx, lb=-20, ub=20, name="x_bar")
        u = m.addVars(k_pred, nu, lb=-1.5, ub=1.5, name="u")
        lam = m.addVars(k_pred, nlam, lb=0, name='Lambda')
        mu = m.addVars(k_pred, nu, lb=-gp.GRB.INFINITY, name='mu')

        # P_term = 2 * P
        obj = (gp.quicksum(P * x_bar[k, i]**2 for k in range(1, k_pred + 1) for i in range(nx)) +
               gp.quicksum(P_mat[i,j] * x_bar[k_pred, i] * x_bar[k_pred, j] for i in range(nx) for j in range(nx)) +
               gp.quicksum(Q * u[k, i]**2 for k in range(k_pred) for i in range(nu)))
        m.setObjective(obj, sense=gp.GRB.MINIMIZE)

        for i in range(nx):
            m.addConstr(x[0, i] == x_atual[i])

        for k in range(k_pred):
            m.addConstr(x[k + 1, 0] == x[k, 0] + x[k, 2] * dt)
            m.addConstr(x[k + 1, 1] == x[k, 1] + x[k, 3] * dt)
            m.addConstr(x[k + 1, 2] == x[k, 2] + u[k, 0] * dt)
            m.addConstr(x[k + 1, 3] == x[k, 3] + u[k, 1] * dt)
            
            m.addConstr(gp.quicksum(-lam[k, j] * b[j] for j in range(nlam)) + 
                        mu[k, 0] * x[k, 0] + mu[k, 1] * x[k, 1] >= 1)

            for i in range(nu):
                m.addConstr(gp.quicksum(lam[k, j] * A[j, i] for j in range(nlam)) - mu[k, i] == 0)

            m.addQConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nu)) <= 1)

        for k in range(k_pred + 1):
            for i in range(nx):
                m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])

        distancia_atual = np.linalg.norm(x_atual[0:2] - x_goal[0:2])
        if distancia_atual < 1.5:
            m.addQConstr(x_bar[k_pred, 0]**2 + x_bar[k_pred, 1]**2 <= 0.25**2)
            m.addQConstr(x_bar[k_pred, 2]**2 + x_bar[k_pred, 3]**2 <= 0.15**2)

        t0 = time.perf_counter()
        m.optimize()
        tf = time.perf_counter()

        if m.status == gp.GRB.OPTIMAL:
            u_aplicar = np.array([u[0, 0].X, u[0, 1].X])
            u_traj_real[t, :] = u_aplicar

            telemetria_passos.append({
                'passo_t': t,
                'k_pred': k_pred,
                'tempo_solver_s': m.Runtime,
                'tempo_wall_s': tf - t0
            })
            
            x_traj_real[t + 1, 0] = x_traj_real[t, 0] + x_traj_real[t, 2] * dt
            x_traj_real[t + 1, 1] = x_traj_real[t, 1] + x_traj_real[t, 3] * dt
            x_traj_real[t + 1, 2] = x_traj_real[t, 2] + u_aplicar[0] * dt
            x_traj_real[t + 1, 3] = x_traj_real[t, 3] + u_aplicar[1] * dt
        else:
            print(f"Falhou para k_pred={k_pred} no passo {t}")
            return None, None, t, None

    return x_traj_real, u_traj_real, passos_executados, telemetria_passos

# --- Parâmetros ---
T = 25
dt = 0.5
nx = 4
nu = 2
nlam = 4

P = 10
Q = 10

P_mat = np.array([
    [45.7230,  0.0000, 30.8341,  0.0000],
    [ 0.0000, 45.7230,  0.0000, 30.8341],
    [30.8341,  0.0000, 55.0745,  0.0000],
    [ 0.0000, 30.8341,  0.0000, 55.0745]
])

A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = np.array([1, 1, 1, 1])

x_init = np.array([-5.0, 0.0, 0.0, 0.0])
x_goal = np.array([5.0, 0.0, 0.0, 0.0])

# Lista de horizontes a testar
lista_k_pred = [3, 5, 7, 10, 12]

# Dicionários e listas para armazenar os dados de todos os k_pred
trajetorias = {}
resultados_estudo = []

for kp in lista_k_pred:
    print(f"Simulando NMPC para k_pred = {kp}...")
    estados, acoes, passos, telemetria = MPC(x_init, k_pred=kp)
    
    if estados is not None:
        trajetorias[kp] = estados  
        
        df_tel = pd.DataFrame(telemetria)
        
        resumo_kp = pd.DataFrame([{
            'k_pred': kp,
            'tempo_medio_passo_ms': df_tel['tempo_solver_s'].mean() * 1000,
            'tempo_total_solver_s': df_tel['tempo_solver_s'].sum(),
            'passos_executados': passos
        }])
        
        # --- SALVAMENTO INDIVIDUAL POR K_PRED ---
        df_tel.to_csv(f"telemetria_passos_k{kp}.csv", index=False)
        resumo_kp.to_csv(f"resumo_desempenho_k{kp}.csv", index=False)
        
        resultados_estudo.append(resumo_kp.iloc[0].to_dict())

df_resumo = pd.DataFrame(resultados_estudo)
df_resumo.to_csv("resumo_geral_todos_k.csv", index=False)
print("\n--- Resumo Final Salvo com Sucesso ---")
print(df_resumo.to_string(index=False))

fig1, ax1 = plt.subplots(figsize=(9, 8))

# Plota cada trajetória armazenada no dicionário
for kp, traj in trajetorias.items():
    ax1.plot(traj[:, 0], traj[:, 1], 'o-', linewidth=2, label=f'$k_{{pred}} = {kp}$')

ax1.plot(x_init[0], x_init[1], 'gs', markersize=10, label='Inicial')
ax1.plot(x_goal[0], x_goal[1], 'rs', markersize=10, label='Objetivo')

square = patches.Rectangle((-1, -1), 2, 2, edgecolor='black', facecolor='red', alpha=0.3, label='Obstáculo')
ax1.add_patch(square)

ax1.set_xlabel('Coordenada X (m)')
ax1.set_ylabel('Coordenada Y (m)')
ax1.set_title('Impacto do Horizonte $k_{pred}$ na Trajetória Realizada')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.legend()
plt.tight_layout()
plt.show()

# FIGURA 2: Custo Computacional vs k_pred
fig2, ax_left = plt.subplots(figsize=(10, 5))

# Tempo Médio por Passo
line1 = ax_left.plot(df_resumo['k_pred'], df_resumo['tempo_medio_passo_ms'], 'bo-', linewidth=2, label='Tempo Médio / Passo (ms)')
ax_left.axhline(y=dt * 1000, color='r', linestyle='--', label=f'Limite Tempo Real ({int(dt*1000)} ms)')
ax_left.set_xlabel('Horizonte de Predição ($k_{pred}$)')
ax_left.set_ylabel('Tempo por Passo (ms)', color='b')
ax_left.tick_params(axis='y', labelcolor='b')
ax_left.grid(True)

# Tempo Total do Solver
ax_right = ax_left.twinx()
line2 = ax_right.plot(df_resumo['k_pred'], df_resumo['tempo_total_solver_s'], 's--', color='green', linewidth=2, label='Tempo Total acumulado (s)')
ax_right.set_ylabel('Tempo Total Solver (s)', color='green')
ax_right.tick_params(axis='y', labelcolor='green')

lines = line1 + [ax_left.get_lines()[1]] + line2
labels = [l.get_label() for l in lines]
ax_left.legend(lines, labels, loc='upper left')

plt.title('Análise de Escalabilidade Computacional do NMPC vs $k_{pred}$')
plt.tight_layout()
plt.show()