from typing import Literal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from utils import set_seed


def create_graph():
    """Creates the graph for the particle dynamics."""
    G = nx.DiGraph()
    nodes = ["o", "a", "b", "c", "d"]
    G.add_nodes_from(nodes)
    edges = [
        ("o", "a"),
        ("o", "b"),
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("c", "d"),
    ]
    mapping = {node: idx for idx, node in enumerate(nodes)}
    G.add_edges_from(edges)
    pos = {
        "o": (0, 2),
        "a": (1, 3),
        "b": (1, 1),
        "c": (2, 1),
        "d": (2, 3),
    }
    Lambda = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1 / 4, 1 / 4, 2 / 4],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    omega = Lambda.sum(axis=1)
    omega[mapping["d"]] = 7 / 4     # add a self-loop in d
    nx.draw(G, pos, with_labels=True, node_color="lightblue", arrowsize=20)
    plt.title("Particle Dynamics Graph")
    plot_path = Path(__file__).parent / "plots" / "Ex3"
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path / "graph.png")
    return G, Lambda, omega, nodes, mapping


def simulate_node_dynamics(
    LAMBDA,
    omega,
    input_lambda,
    nodes,
    mapping,
    max_time,
    rate_type: Literal["proportional", "fixed"] = "proportional",
):
    """Simulates the node dynamics for a given input lambda."""
    D = np.diag(omega)
    P = np.linalg.inv(D) @ LAMBDA
    events = ["input"] + nodes                                # include the input event (a new particle spawns)
    n_events = len(events)

    particle_counts = np.zeros(len(nodes), dtype=int)
    proportional_rate = rate_type == "proportional"           # True if rate_type is proportional
    history = [particle_counts.copy()]                        # initialize the historical storage
    times = [0]
    total_time = 0

    if proportional_rate:
        activation = particle_counts                        # if proportional_rate, coefficients are the particles' counters
    else:
        activation = particle_counts > 0                    # if not proportional_rate, coefficients are 1 if particles are present and 0 otherwise
    
    rates = np.zeros(n_events)
    rates[0] = input_lambda                                 # set the rate of the input event
    rates[1:] = omega * activation                          # set the transitions' rates
    global_rate = np.sum(rates)                             # initialize the global clock's rate
    
    while total_time < max_time:
        t_next = -np.log(np.random.rand()) / global_rate
        total_time += t_next
        next_node = np.random.choice(                       # select the next event that happens
            events,                 
            p = rates / global_rate,
        )

        if next_node == "input":                            
            particle_counts[mapping["o"]] += 1              # if input, spawn a new particle in node 'o'
        else:
            idx = mapping[next_node]
            particle_counts[idx] -= 1                       # remove a particle from next_node
            if idx != mapping["d"]:
                destination = np.random.choice(nodes, p=P[idx, :])     # select the destination if next_node != 'd'
                particle_counts[mapping[destination]] += 1             # add the moving particle to the destination

        # Update global rate and probabilities
        if proportional_rate:
            activation = particle_counts
        else:
            activation = particle_counts > 0
        
        rates[1:] = omega * activation                      # transition rates only must be updated
        global_rate = np.sum(rates)                         # global rate updates too
        history.append(particle_counts.copy())
        times.append(total_time)

    return np.array(history), np.array(times)


def plot_stacked_area(history, times, nodes, plot_dir, title):
    """Plots the stacked area chart of particle counts over time."""
    plt.figure(figsize=(10, 6))
    plt.stackplot(times, history.T, labels=nodes)
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Particles")
    plt.title(title)
    plt.legend(loc="upper left")
    plot_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir)
    plt.show()

def print_main_header():
    print("\n" + "=" * 80)
    print("ESERCIZIO 3, PARTICLE DYNAMICS IN OPEN NETWORK")
    print("=" * 80 + "\n")


def print_case_header(title, lambda_, max_time, rate_type):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    print(f"  • Rate type      : {rate_type}")
    print(f"  • Lambda (λ)     : {lambda_:.2f}")
    print(f"  • Max simulation : {max_time} (time units)")
    print("-" * 80 + "\n")


def plot_stacked_area(history, times, nodes, plot_path, title):
    """Plots the stacked area chart of particle counts over time."""
    plt.figure(figsize=(10, 6))
    plt.stackplot(times, history.T, labels=nodes)
    plt.xlabel("Time")
    plt.ylabel("Number of Particles")
    plt.title(title)
    plt.legend(loc="upper left")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    set_seed(42)
    
    G, LAMBDA, omega, nodes, mapping = create_graph()

    # Header generale
    print_main_header()
    print("Graph and transition matrix fixed for all the experiments.\n")
    print("Transition Rate Matrix (Λ):")
    print(LAMBDA)
    print("\nOutgoing Rates (ω):")
    print(omega)
    print("\nNodes:", nodes)
    print("=" * 80)

    # ==============================================================
    # 1) PROPORTIONAL RATES – λ = 100, max_time = 60
    # ==============================================================

    input_lambda = 100.0
    max_time = 60

    print_case_header(
        "CASO 1 – PROPORTIONAL RATES",
        input_lambda,
        max_time,
        rate_type="proportional"
    )

    history, times = simulate_node_dynamics(
        LAMBDA,
        omega,
        input_lambda,
        nodes,
        mapping,
        max_time,
        rate_type="proportional",
    )

    plot_path = (
        Path(__file__).parent
        / "plots"
        / "Ex3"
        / "proportional_rate_dynamics_lambda_100.png"
    )

    plot_stacked_area(
        history,
        times,
        nodes,
        plot_path,
        title="Particle Counts Over Time – Proportional Rates (λ = 100)"
    )

    print("Shape history           :", history.shape)
    print("Mean particles per node :", np.round(history.mean(axis=0), 3))
    print("Expected theoretical    :", [50, 50, 500 / 8, 300 / 4, 400 / 7])
    print("-" * 80)

    # ==============================================================
    # 2) FIXED RATES – λ = 2.00, max_time = 6000 (unstable case)
    # ==============================================================

    input_lambda = 2.0
    max_time = 6000

    print_case_header(
        "CASO 2 – FIXED RATES (unstable case)",
        input_lambda,
        max_time,
        rate_type="fixed"
    )

    history, times = simulate_node_dynamics(
        LAMBDA,
        omega,
        input_lambda,
        nodes,
        mapping,
        max_time,
        rate_type="fixed",
    )

    plot_path = (
        Path(__file__).parent
        / "plots"
        / "Ex3"
        / "fixed_rate_lambda_2.00_dynamics.png"
    )

    plot_stacked_area(
        history,
        times,
        nodes,
        plot_path,
        title="Particle Counts Over Time – Fixed Rates (λ = 2.00, unstable)"
    )

    print("Shape history           :", history.shape)
    print("Mean particles per node :", np.round(history.mean(axis=0), 3))
    print("-" * 80)

    # ==============================================================
    # 3) FIXED RATES – λ = 1.32, max_time = 6000 (critical case)
    # ==============================================================

    input_lambda = 1.32

    print_case_header(
        "CASO 3 – FIXED RATES (critical case)",
        input_lambda,
        max_time,
        rate_type="fixed"
    )

    history, times = simulate_node_dynamics(
        LAMBDA,
        omega,
        input_lambda,
        nodes,
        mapping,
        max_time,
        rate_type="fixed",
    )

    plot_path = (
        Path(__file__).parent
        / "plots"
        / "Ex3"
        / "fixed_rate_lambda_1.32_dynamics.png"
    )

    plot_stacked_area(
        history,
        times,
        nodes,
        plot_path,
        title="Particle Counts Over Time – Fixed Rates (λ = 1.32, critical)"
    )

    print("Shape history           :", history.shape)
    print("Mean particles per node :", np.round(history.mean(axis=0), 3))
    print("-" * 80)

