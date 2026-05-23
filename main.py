import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

T = 10
nxu = 2
nlam = 4

P = np.eye(nxu)
Q = np.eye(nxu)

A = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
])
b = np.array([1,1,1,1])

x_init = [-5, 0]
x_goal = [5, 0]
m = gp.Model("NMPC")

m.setParam("NonConvex", 2)


x = m.addVars(T, nxu, lb=-6, ub=6, name="pos")
x_bar = m.addVars(T, nxu, lb=-20, ub=20, name="x_bar")
u = m.addVars(T, nxu, lb=-1, ub=1, name="u")
lam = m.addVars(T, nlam, lb=0, name='Lambda')
mu = m.addVars(T, nxu, name='mu')

obj = gp.quicksum(P[i,i] * x_bar[k,i]**2 for k in range(T) for i in range(nxu)) + gp.quicksum(Q[i,i] * u[k, i]**2 for k in range(T) for i in range(nxu))
m.setObjective(obj, sense=gp.GRB.MINIMIZE)

for i in range(nxu):
    m.addConstr(x[0,i] == x_init[i] + u[0,i])
    m.addConstr(x[T-1, i] == x_goal[i])

for k in range(T):
    for i in range(nxu):
        m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])

for k in range(1, T):
    for i in range(nxu):
        m.addConstr(x[k, i] == x[k-1,i] + u[k, i])


    m.addConstr(-gp.quicksum(lam[k, l] * b[l] for l in range(nlam) ) + gp.quicksum(mu[k, i] * x[k, i]  for i in range(nxu)) >= 1)

    for i in range(nxu):
        m.addConstr(gp.quicksum(lam[k, l] * A[l, i] for l in range(nlam)) - mu[k, i] == 0)


    m.addQConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nxu)) <= 1)

m.optimize()

if m.status == gp.GRB.OPTIMAL:
    print("Ótimo encontrado")
else:
    print("Status:", m.status)


x_traj = []
x_traj.append(x_init)
x_traj.extend([x[k,0].X, x[k,1].X] for k in range(T))

x_traj = np.array(x_traj)

fig , ax = plt.subplots(figsize=(8,8))

ax.plot(
    x_traj[:,0],
    x_traj[:,1],
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