import networkx as nx
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from utils import save_image,simulate_FDG
from scipy.stats import norm

# --------------------------------------------------------------
# ------ Creation and visualization of the graph ---------------
# --------------------------------------------------------------

path_ex1 = 'Homework_2/plots/Ex1'
visualization = False                           # Flag to visualize plots (if True)
separator = '-' * 50

nodes = ['o','a','b','c','d']
LAMBDA = [[0,2/5,1/5,0,0],
          [0,0,3/4,1/4,0],
          [1/2,0,0,1/3,0],
          [0,0,1/3,0,2/3],
          [0,1/3,0,1/3,0]]

edges = [   ('o','a', {'weight' : 2/5}),
            ('o','b', {'weight' : 1/5}),
            ('a','b', {'weight' : 3/4}),
            ('a','c', {'weight' : 1/4}),  
            ('b','o', {'weight' : 1/2}),
            ('b','c', {'weight' : 1/3}),
            ('c','b', {'weight' : 1/3}),
            ('c','d', {'weight' : 2/3}),
            ('d','a', {'weight' : 1/3}),
            ('d','c', {'weight' : 1/3})]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

pos_init = {
    'o': (-1.5,  0.0),
    'a': ( 0.0,  1.0),
    'b': ( 0.0, -1.0),
    'c': ( 1.7, -0.3),
    'd': ( 1.7,  0.9),
}

nx.draw_networkx_nodes(G,pos_init, node_color='lightblue', node_size=1000)
nx.draw_networkx_labels(G,pos_init,font_color='b',font_weight='black',font_size=10)

for u,v in G.edges():
        rad = 0.10
        if G.has_edge(v,u):
            rad = 0.35

        nx.draw_networkx_edges(
            G, pos_init,
            edgelist=[(u,v)],
            connectionstyle=f"arc3, rad={rad}",
            arrowsize=18,
            arrowstyle='-|>',
            min_source_margin=12,
            min_target_margin=12,
            width=1.4
        )

name = '/graph_ex_1'
save_image(path_ex1 + name,visualization=visualization)

# --------------------------------------------------------------
# ------------------ Simulation Setting ------------------------
# --------------------------------------------------------------

np.random.seed(26)

omega = np.sum(LAMBDA, axis=1)                     # Already real
D = np.diag(omega)
P = np.linalg.inv(D) @ LAMBDA

map = { 0 : 'o',
        1 : 'a',
        2 : 'b',
        3 : 'c',
        4 : 'd'
    }

n_steps = 1000                                  # Len simulation
n_states = P.shape[0]                           
pos = np.zeros(n_steps)                         # Memorizes trajectory, i.e. where simulation is at current time
pos[0] = 0                                      # We start from node o
transition_times = np.zeros(n_steps)            # Stores instants in which simulation jumps from a state to another

t_next = -np.log(np.random.rand())/omega[0]         

for i in range(1,n_steps):
    pos[i] = choice(n_states,p=P[int(pos[i-1]),:])          # choise randomly where to go form previous state based on probabilities
    transition_times[i] = transition_times[i-1] + t_next        # store when will has been made jump
    t_next = -np.log(np.random.rand())/omega[int(pos[i])]       # find new tic based on rate of clock current state (in which we just did jump)

pi_bar = np.zeros(n_states)
intervals = np.diff(transition_times,n=1,append=(transition_times[-1] + t_next))

for node in range(n_states):
    visits = np.argwhere(pos==node)
    pi_bar[node] = np.sum(intervals[visits])/(transition_times[-1] + t_next)

state_colors = {
    0: 'tab:blue',
    1: 'tab:orange',
    2: 'tab:green',
    3: 'tab:red',
    4: 'tab:purple'
}

colors = [state_colors[int(s)] for s in pos[:30]]

plt.scatter(transition_times[:30], pos[:30], c=colors, s=80)
plt.yticks([0,1,2,3,4], ['o','a','b','c','d'])
plt.title("Trajectory for the first 30 jumps")

name = '/trajectory'
save_image(path_ex1 + name,visualization)


# --------------------------------------------------------------
# ---------------- (a)Simulated Return Time --------------------
# --------------------------------------------------------------

idx_a = np.argwhere([el == 1 for el in pos])
times_a = transition_times[idx_a].flatten()
a_visits = len(times_a)

last = True if pos[-1] == 1 else False              # Flag is True if last element of simulation is an a
                                                    # and we should sum an artificial timestamp

if last:
    t_next_a = -np.log(np.random.rand())/omega[1]           # 1 is mapping of 'a'
    intervals_a = np.diff(times_a,n=1,append=(times_a[-1] + t_next_a))
else:
    intervals_a = np.diff(times_a,n=1)

exp_avg_return_time_a = np.sum(intervals_a)/a_visits
print(separator)
print('POINT A')
print(f'Expected Return Time(a) = {exp_avg_return_time_a}')
print(separator)
print()


# --------------------------------------------------------------
# ---------------- (b)Theoretical Return Time ------------------
# --------------------------------------------------------------

S = [1]
R = [node for node in range(n_states) if node not in S]
beta = 1/omega[R]
m = len(R)

hat_P = P[np.ix_(R,R)]                                          # P restricted to R
hat_x = np.linalg.solve((np.identity(m) - hat_P),beta)

hittings_s = np.zeros(n_states)
for p,r in enumerate(R):
    hittings_s[r] = hat_x[p]

print(separator)
print('POINT B')
print(f'Theoretical hitting times to reach a: {hittings_s}')        # As we know, hitting time of 'a' is 0

return_time_a = 1/omega[1] + np.dot(P[1,:],hittings_s)

print(f'Theoretical return time of a: {return_time_a}')
print(f'Gap estimation = {abs(return_time_a - exp_avg_return_time_a)}')
print(separator)
print()


# --------------------------------------------------------------
# ------------------- (c)o-d Hitting Time ----------------------
# --------------------------------------------------------------

intervals_od = []
start_time = None                                                    

for t, state in zip(transition_times, pos):
    # If in o and no interval is measuring, open a window 
    if state == 0:
        if start_time is None:
            start_time = t

    # If in d and interval is measuring, close the window
    elif state == 4 and start_time is not None:
        intervals_od.append(t - start_time)
        start_time = None                                   # Wait until next o occures

intervals_od = np.array(intervals_od)

exp_avg_hitting_time_od = intervals_od.mean()

print(separator)
print('POINT C')
print(f"Expected Hitting Time(d) from o = {exp_avg_hitting_time_od}")
print(separator)
print()


# --------------------------------------------------------------
# ---------------- (d)Theoretical Hitting Time -----------------
# --------------------------------------------------------------

d_S = [4]
d_R = [node for node in range(n_states) if node not in d_S]
d_beta = 1/omega[d_R]
d_m = len(d_R)

hat_d_P = P[np.ix_(d_R,d_R)]
hat_d_x = np.linalg.solve((np.identity(d_m) - hat_d_P),d_beta)

d_hittings_s = np.zeros(n_states)
for p,r in enumerate(d_R):
    d_hittings_s[r] = hat_d_x[p]

o_hitting_d = d_hittings_s[0]

print(separator)
print('POINT D')
print(f'Theoretical hitting time to reach d from o: {o_hitting_d}')
print(f'Gap estimation = {abs(o_hitting_d - exp_avg_hitting_time_od)}')
print(separator)
print()

# --------------------------------------------------------------
# ---------------- (e)French-DeGroot Dynamics ------------------
# --------------------------------------------------------------

print(separator)
print('POINT E')
print(f'Graph is Aperiodic: {nx.is_aperiodic(G)}')

np.random.seed(26)
n_it = 100
x0 = np.random.rand(n_states)
x = simulate_FDG(n_it,n_states,P,x0)      # Defined in utils.py

means = x.mean(axis=1)                    # Estimates of alpha at each iteration t
variances = x.var(axis=1)
estimated_alpha = means[-1]               #This is an estimate of consensun since we are averaging, 
                                          #meaning that single node could not have reached consensus in this confguration

tol = 1e-6
it_to_converge = np.where(np.abs(means-estimated_alpha) < tol)[0][0]

fig,axs = plt.subplots(1,2,figsize=(12,5))
axs[0].plot(range(n_it), means)
axs[0].set_xlabel("t")
axs[0].set_ylabel("mean opinion")
axs[1].plot(range(n_it), x.std(axis=1))
axs[1].set_xlabel("t")
axs[1].set_ylabel("std opinion")
plt.autoscale(enable=True, axis='y', tight=True)

print(f'Average opinion: {estimated_alpha}, reached after {it_to_converge} iteration')
print(f'Mean variance: {np.mean(variances)}')

name = '/average_opinion_and_variance'
save_image(path_ex1 + name, visualization=visualization)


M = 1000                                # Number of Mote Carlo iterations
estimated_alpha_error = np.zeros(M)

for i in range(M):
    x0 = np.random.rand(n_states)
    x = simulate_FDG(n_it,n_states,P,x0)

    est_alpha = np.mean(x[-1])
    alpha = pi_bar @ x0
    estimated_alpha_error[i] = np.mean(np.abs(est_alpha - alpha))

print(f'Average error of estimation : {np.mean(estimated_alpha_error)}')
print(separator)
print()

# --------------------------------------------------------------
# -------------- (f)Variance of Consensus Value ----------------
# --------------------------------------------------------------

var_alpha = 2*(pi_bar[1]**2 + pi_bar[2]**2 + pi_bar[3]**2) + (pi_bar[0]**2 + pi_bar[4]**2)
print(separator)
print('POINT F')
print(f'Theoretical variance of alpha = {var_alpha}')
print(f'Variance of alpha less than variances of individuals: {all(var_alpha < variance for variance in variances)}')


variances = [1,2,2,2,1]                  
alphas = np.zeros(M)

for i in range(M):
    x0 = np.zeros(n_states)
    
    for k in range(n_states):
        x0[k] = np.random.normal(0,np.sqrt(variances[k]))
    
    alphas[i] = pi_bar @ x0

est_var_alpha = np.var(alphas)

# Theoretical Gaussian parameters
mean_theo = 0
std_theo = np.sqrt(var_alpha)

# Empirical Gaussian parameters
mean_emp = np.mean(alphas)
std_emp = np.std(alphas)

plt.figure(figsize=(8,5))

# Histogram of Monte Carlo samples (empirical distribution)
plt.hist(alphas, bins=40, density=True, alpha=0.4,
         label="Empirical distribution")

xmin, xmax = min(alphas), max(alphas)
x = np.linspace(xmin, xmax, 400)

# Theoretical Gaussian
plt.plot(x, norm.pdf(x, mean_theo, std_theo),
         'r-', linewidth=2,
         label=f"Theoretical Gaussian (σ={std_theo:.4f})")

# Empirical Gaussian
plt.plot(x, norm.pdf(x, mean_emp, std_emp),
         'k--', linewidth=2,
         label=f"Empirical Gaussian (μ={mean_emp:.4f}, σ={std_emp:.4f})")

plt.xlabel("Consensus value α")
plt.ylabel("Density")
plt.title("Comparison between theoretical and empirical distribution of α")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.3)

name = '/alpha_gaussians'
save_image(path_ex1 + name, visualization=visualization)

print(f'Estimated variance of alpha = {est_var_alpha}')
print(f'Gap from estimated alpha variance = {np.abs(var_alpha - est_var_alpha)}')
print(separator)
print()

# --------------------------------------------------------------
# ----------- (g)Asymptotic behaviour removing edges -----------
# --------------------------------------------------------------

print(separator)
print('POINT G')

# Fixed initialization for point (g): keep {o,a,b,d} constant across all experiments
np.random.seed(26)
x0_g = np.random.rand(n_states)

mapping = {node: idx for idx, node in map.items()}

G_g = G.copy()
edges_to_remove = [('d', 'a'), ('d', 'c'), ('a', 'c'), ('b', 'c')]
G_g.remove_edges_from(edges_to_remove)

# Add self loop with weight 1 on d as sink node
G_g.add_edge('d', 'd', weight=1.0)

nx.draw(G_g, pos_init, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, font_weight='bold', arrowsize=15)

name = '/graph_point_g'
save_image(path_ex1+name,visualization=visualization)

LAMBDA_g = np.array(LAMBDA, dtype=float).copy()

for start, end in edges_to_remove:
    i = mapping[start]
    j = mapping[end]
    LAMBDA_g[i, j] = 0.0

# self-loop on d
idx_d = mapping['d']
LAMBDA_g[idx_d, idx_d] = 1.0

omega_g = LAMBDA_g.sum(axis=1)
D_g = np.diag(omega_g)
P_g = np.linalg.inv(D_g) @ LAMBDA_g

H = np.array([
    [1.0, 0.0],                 # node 'o'
    [1.0, 0.0],                 # node 'a'
    [1.0, 0.0],                 # node 'b'
    [1.0/3.0, 2.0/3.0],         # node 'c'
    [0.0, 1.0]                  # node 'd'
])

x_bar0 = 3.0/8.0 * x0_g[0] + 1.0/4.0 * x0_g[1] + 3.0/8.0 * x0_g[2]
x_bar1 = x0_g[4]
x_bar = np.array([x_bar0, x_bar1])

x_bar_0 = x_bar
x_theoretical_inf = H @ x_bar_0

print(f"Theoretical asymptotic opinion vector: {x_theoretical_inf}")

n_it_g = 1000
x_g = simulate_FDG(n_it_g, n_states, P_g, x0_g)
x_sim_inf = x_g[-1, :]

print(f"Simulated asymptotic opinion vector after {n_it_g} iterations: {x_sim_inf}")
print(f"Gap between theoretical and simulated limit: {np.linalg.norm(x_theoretical_inf - x_sim_inf)}")


# --------------------------------------------------------------
# ---- (g-extra) Behaviour of the three components (plots) -----
# --------------------------------------------------------------

# 1) Evolution of three components over time
n_it_comp = 50
x_traj_g = simulate_FDG(n_it_comp, n_states, P_g, x0_g)

comp_oab = x_traj_g[:, [0, 1, 2]].mean(axis=1)              # Internal Equilibrium of {o,a,b}
comp_c = x_traj_g[:, 3]
comp_d = x_traj_g[:, 4]

plt.figure(figsize=(6,4))
plt.plot(range(n_it_comp), comp_oab, label='component {o,a,b}')
plt.plot(range(n_it_comp), comp_c,  label='node c')
plt.plot(range(n_it_comp), comp_d,  label='node d')
plt.xlabel('t')
plt.ylabel('opinion')
plt.title('Evolution of components under modified graph $G_g$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

name = '/components_evolution_point_g'
save_image(path_ex1 + name, visualization=visualization)


# 2) Independency of limit value of x from its initial opinion
n_it_eq = 100
n_sim   = 50           # number of different initializations of c

base_x0 = x0_g.copy()
c_inits = np.linspace(0.0, 1.0, n_sim)
c_finals = np.zeros(n_sim)

for i, c0 in enumerate(c_inits):
    x0_tmp = base_x0.copy()
    x0_tmp[3] = c0                                              # only c changes
    x_traj = simulate_FDG(n_it_eq, n_states, P_g, x0_tmp)
    c_finals[i] = x_traj[-1, 3]                                 # c asyntotic value

plt.figure(figsize=(6,4))
plt.plot(c_inits, c_finals)
plt.xlabel(r'initial value $x_c(0)$')
plt.ylabel(r'limit value $x_c(\infty)$')
plt.title('Value of $c$ at equilibrium vs initialisation')
plt.grid(True, linestyle='--', alpha=0.3)

name = '/c_equilibrium_vs_initialisation_point_g'
save_image(path_ex1 + name, visualization=visualization)


x_bar0 = 3.0/8.0 * base_x0[0] + 1.0/4.0 * base_x0[1] + 3.0/8.0 * base_x0[2]
x_bar1 = base_x0[4]
x_bar_base = np.array([x_bar0, x_bar1])
x_c_inf_theo = H[3, :] @ x_bar_base

print(f"Theoretical limit of c (from H, base_x0): {x_c_inf_theo}")
print(f"Average simulated limit of c over different x_c(0): {c_finals.mean()}")
print(separator)
print()

# --------------------------------------------------------------
# --------------- (h)Behaviour on another graph ----------------
# --------------------------------------------------------------

print(separator)
print('POINT H')

G_h = G.copy()
edges_to_remove_h = [('b','o'), ('d','a')]
G_h.remove_edges_from(edges_to_remove_h)

nx.draw(G_h, pos_init, with_labels=True,
        node_color='lightblue', node_size=1000,
        font_size=10, font_weight='bold', arrowsize=15)

name = '/graph_point_h'
save_image(path_ex1 + name, visualization=visualization)

LAMBDA_h = np.array(LAMBDA, dtype=float).copy()

for start, end in edges_to_remove_h:
    i = mapping[start]     
    j = mapping[end]
    LAMBDA_h[i, j] = 0.0   

omega_h = LAMBDA_h.sum(axis=1)
D_h = np.diag(omega_h)
P_h = np.linalg.inv(D_h) @ LAMBDA_h

print(f"Aperiodicity of sink {{b,c,d}}: {nx.is_aperiodic(G_h.subgraph(['b','c','d']))}")

np.random.seed(26)
x0_h = np.random.rand(n_states)
x_h  = simulate_FDG(n_it, n_states, P_h, x0_h)

print()
print("Last two opinion vectors (t = n_it-2, n_it-1):")
print("x_h[n_it-2] =", x_h[-2])
print("x_h[n_it-1] =", x_h[-1])
print("Difference:",x_h[-1] - x_h[-2])
print()

means_over_time = x_h.mean(axis=1)
print(f"Simulated asymptotic opinion vector after {n_it} iterations: {x_h[-1]}")

node_time_averages = x_h.mean(axis=0)
for i in range(n_states):
    print(f"Time-average of node {map[i]}: {node_time_averages[i]:.4f}")

plt.figure(figsize=(8,4))
plt.plot(range(n_it), means_over_time)
plt.xlabel('t')
plt.ylabel('mean')
plt.title('Time Average under modified graph $G_h$')
plt.grid(True, linestyle='--', alpha=0.3)

name = '/time_average_point_h'
save_image(path_ex1 + name, visualization=visualization)

plt.figure(figsize=(8,4))

for i in range(n_states):
    plt.plot(range(10), x_h[:10, i],
             c=state_colors[i],
             label=map[i])

plt.xlabel('t')
plt.ylabel('opinion')
plt.title('States Equilibria')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

name = '/states_equilibria_point_h'
save_image(path_ex1 + name, visualization=visualization)

print(separator)
print()