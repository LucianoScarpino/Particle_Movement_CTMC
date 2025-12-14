import random
from typing import Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed_value: int) -> None:
    """Set the seed for random number generators in random and numpy."""
    random.seed(seed_value)
    np.random.seed(seed_value)


def progress_bar(initial_string, current, total, bar_length=100):
    """Prints a progress bar to the console, with no required additional libraries."""

    fraction = current / total  # Calculate the fraction completed
    arrow = int(fraction * bar_length) * "="  # Create the arrow part of the bar
    padding = int(bar_length - len(arrow)) * "."  # Create the padding part of the bar
    ending = "\n" if current == total else "\r"  # Determine if we end the line or return carriage

    # Print the progress bar
    print(f"{initial_string}\t [{arrow}{padding}] {int(fraction*100)}%", end=ending)


def create_ex2_graph(plot_dir: Optional[Path] = None):
    """Creates the directed graph for Exercise 2 along with its transition rate matrix.
    Args:
        plot_dir (Optional[Path]): If provided, saves a plot of the graph to this path. It also creates parent directories if they do not exist.
    Returns:
        G (nx.DiGraph): The directed graph.
        Lambda (np.ndarray): The transition rate matrix.
        nodes (list): List of node labels.
    """

    # Define nodes
    nodes = ["o", "a", "b", "c", "d"]

    # Define transition rate matrix
    Lambda = np.array(
        [
            [0, 2 / 5, 1 / 5, 0, 0],
            [0, 0, 3 / 4, 1 / 4, 0],
            [1 / 2, 0, 0, 1 / 3, 0],
            [0, 0, 1 / 3, 0, 2 / 3],
            [0, 1 / 3, 0, 1 / 3, 0],
        ]
    )

    # Define edges with weights
    edges = [
        ("o", "a", {"weight": 2 / 5}),
        ("o", "b", {"weight": 1 / 5}),
        ("a", "b", {"weight": 3 / 4}),
        ("a", "c", {"weight": 1 / 4}),
        ("b", "o", {"weight": 1 / 2}),
        ("b", "c", {"weight": 1 / 3}),
        ("c", "b", {"weight": 1 / 3}),
        ("c", "d", {"weight": 2 / 3}),
        ("d", "a", {"weight": 1 / 3}),
        ("d", "c", {"weight": 1 / 3}),
    ]

    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Plot the graph
    pos_init = {
        "o": (-1.5, 0.0),
        "a": (0.0, 1.0),
        "b": (0.0, -1.0),
        "c": (1.7, -0.3),
        "d": (1.7, 0.9),
    }
    nx.draw(
        G,
        pos_init,
        with_labels=True,
        node_color="lightblue",
        node_size=1000,
        font_size=10,
        font_weight="bold",
        arrowsize=20,
    )

    # Save plot if directory is provided
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir)
        plt.show()
    
    plt.clf()

    omega = Lambda.sum(axis=1)
    omega_star = omega.max()
    P_bar = Lambda / omega_star
    P_bar += np.eye(len(nodes)) * (1 - P_bar.sum(axis=1))
    eigenvalues, eigenvectors = np.linalg.eig(P_bar.T)
    max_index = np.argmax(eigenvalues.real)
    pi_bar = eigenvectors[:, max_index].real
    pi_bar /= pi_bar.sum()  # Normalize to make it a probability distribution

    # Return the graph, transition rate matrix, and nodes
    return G, Lambda, nodes, omega, P_bar, pi_bar

def save_image(plot_dir,visualization=True):
    """ Function to store image in given directory """
    plt.savefig(plot_dir)

    if visualization:
        plt.show()
        
    plt.clf()
    plt.close()

def simulate_FDG(n,n_states,P,x0):
    """ Perform FDG dybamics.
    Args:
        n (int): Number of iterations.
        n_states (int): Number of states in the opinion dynamics.
        P (np.ndarray): Stochastic matrix of transitional probabilities.
        x0 (np.ndarray): Array containing initial opinions.
    """

    x_t = x0
    x = np.zeros((n,n_states))
    for i in range(n):
        x_t1 = P @ x_t
        x[i] = (x_t1)
        x_t = x_t1
    
    return x