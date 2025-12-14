from pathlib import Path
import networkx as nx
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import random

from utils import create_ex2_graph, set_seed, progress_bar
from Ex2 import simulate_particle_return_time


def run_particle_simulation(
    starting_node,
    mapping,
    P_bar,
    omega_star,
    expected_return_time,
    samples,
    number_of_nodes_list,
    n_iterations_list,
    nodes,
    plots_path: Path,
):

    results = np.zeros(
        (samples, len(n_iterations_list), len(number_of_nodes_list))
    )  # samples x iterations x N
    for i, number_of_nodes in enumerate(number_of_nodes_list):
        rate = omega_star * number_of_nodes
        for j, n_iterations in enumerate(n_iterations_list):
            for sample_idx in range(samples):

                # Style points
                progress_bar(
                    f"N={number_of_nodes} \t iterations={n_iterations} \t sample={sample_idx+1}",
                    sample_idx + 1,
                    samples,
                )

                # Run simulation
                results[sample_idx, j, i] = simulate_particle_return_time(
                    starting_node,
                    P_bar,
                    nodes,
                    mapping,
                    rate,
                    number_of_nodes,
                    n_iterations,
                )

    for i, number_of_nodes in enumerate(number_of_nodes_list):
        plt.figure()

        x = np.array(n_iterations_list)
        y_mean = results[:, :, i].mean(axis=0)
        y_min = results[:, :, i].min(axis=0)
        y_max = results[:, :, i].max(axis=0)

        plt.plot(x, y_mean, "-o", label="Mean over samples")
        plt.fill_between(x, y_min, y_max, alpha=0.2, label="Min-Max over samples")

        plt.axhline(
            y=expected_return_time,
            color="r",
            linestyle="--",
            label=f"Theoretical Value ({expected_return_time})",
        )
        plt.xlabel("Number of iterations")
        plt.ylabel("Estimated Average Return Time")
        plt.title(f"Estimated Return Time vs Iterations (N={number_of_nodes})")
        plt.legend()
        plt.savefig(plots_path / f"Ex2_return_time_N_{number_of_nodes}.png")
        plt.clf()


def main():
    set_seed(42)
    G, LAMBDA, nodes, omega, P_bar, pi_bar = create_ex2_graph(
        plot_dir=Path(__file__).parent / "plots" / "Ex2_graph.png"
    )

    # Point (a)
    starting_node = "a"
    omega_star = omega.max()
    mapping = {node: idx for idx, node in enumerate(nodes)}
    expected_return_time = 1 / (
        omega[mapping[starting_node]] * pi_bar[mapping[starting_node]]
    )
    print(
        f"Theoretical expected return time to node '{starting_node}': {expected_return_time}"
    )

    run_particle_simulation(
        starting_node,
        mapping,
        P_bar,
        omega_star,
        expected_return_time,
        samples=20,
        number_of_nodes_list=[1, 10, 100],
        n_iterations_list=[1_000, 10_000, 100_000],
        nodes=nodes,
        plots_path=Path(__file__).parent / "plots",
    )


if __name__ == "__main__":
    main()
