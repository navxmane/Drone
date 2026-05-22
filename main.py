import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt

T = 10
nxu = 2
nlam = 4

P = np.eye(nxu) * 10
Q = np.eye(nxu) * 10

A = np.array([[-1,0], [0,1], [1,0], [0,-1]])
b = np.array([1,1,1,1])

x_init = [-5, 0]
x_goal = [5, 0]
m = gp.Model("NMPC")

x = m.addVars(T+1, nxu, lb=-5, ub=5, name="pos")
x_bar = m.addVars(T +1, nxu, lb=-10, ub=10, name="x_bar")
u = m.addVars(T, nxu, lb=-2, ub=2, name="u")
lam = m.addVars(T, nlam, lb=0, name='Lambda')
mu = m.addVars(T, nxu, name='mu')

obj = gp.quicksum(P[i,i] * x_bar[k,i]**2 for k in range(T) for i in range(nxu)) + gp.quicksum(Q[i,i] * u[k, i]**2 for k in range(T) for i in range(nxu))
m.setObjective(obj, sense=gp.GRB.MINIMIZE)

for i in range(nxu):
    m.addConstr(x[0,i] == x_init[i])
    m.addConstr(x[T, i] == x_goal[i])

for k in range(T):
    for i in range(nxu):
        m.addConstr(x_bar[k, i] == x[k, i] - x_goal[i])
        m.addConstr(x[k+1, i] == x[k,i] + u[k, i])


    m.addConstr(-gp.quicksum(lam[k, l] * b[l] for l in range(nlam) ) + gp.quicksum(mu[k, i] * x[k, i]  for i in range(nxu)) >= 1)

    for i in range(nxu):
        m.addConstr(gp.quicksum(lam[k, l] * A[l, i] for l in range(nlam)) - mu[k, i] == 0)


    m.addConstr(gp.quicksum(mu[k, i] * mu[k, i] for i in range(nxu)) <= 1)

m.optimize()

x_traj = np.array([
    [x[k,0].X, x[k,1].X]
    for k in range(T+1)
])

plt.figure(figsize=(7,7))

plt.plot(
    x_traj[:,0],
    x_traj[:,1],
    'bo-',
    linewidth=2,
    label='Trajetória'
)

plt.plot(
    x_init[0],
    x_init[1],
    'gs',
    markersize=10,
    label='Inicial'
)

plt.plot(
    x_goal[0],
    x_goal[1],
    'rs',
    markersize=10,
    label='Objetivo'
)

plt.xlabel('x')
plt.ylabel('y')

plt.title('Trajetória ótima')

plt.grid(True)
plt.axis('equal')
plt.legend()

plt.show()