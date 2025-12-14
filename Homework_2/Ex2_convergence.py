"""This script is an addendum to Ex2.py, focusing on the convergence of the particle simulation,
comparing estimated return times with theoretical values as the number of particles and iterations/time increase.
It generates plots to visualize convergence behavior."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import create_ex2_graph, set_seed, progress_bar
from Ex2 import simulate_particle_return_time


def run_particle_simulation(
    starting_node,
    mapping,
    P_bar,
    omega_star,
    expected_return_time,
    samples,
    n_particles_list,
    n_iterations_list,
    times_list,
    nodes,
    plots_path: Path,
):
    """
    Runs the particle simulation for different numbers of particles and either
    number of iterations or time limits, and plots the estimated return times.
    It averages results over multiple samples for robustness.
    Args:
        starting_node: The node from which particles start.
        mapping: A mapping from node names to indices.
        P_bar: Transition probability matrix.
        omega_star: Maximum rate for scaling.
        expected_return_time: The theoretical expected return time for comparison.
        samples: Number of samples to average over.
        n_particles_list: List of different numbers of particles to simulate.
        n_iterations_list: List of different numbers of iterations to run (if not None).
        times_list: List of different time limits to run (if not None).
        nodes: List of node names.
        plots_path: Path to save the plots.
    """

    # Sanity checks
    assert n_iterations_list is None or times_list is None, "Provide either n_iterations_list or times_list, not both."
    assert n_iterations_list is not None or times_list is not None, "Provide one of n_iterations_list or times_list." 

    # Initialize results array
    results = np.zeros(
        (samples, len(n_iterations_list or times_list), len(n_particles_list))
    )
    for i, n_particles in enumerate(n_particles_list):
        rate = omega_star * n_particles # Scale rate with number of particles
        if times_list is not None:
            for j, max_time in enumerate(times_list):
                for sample_idx in range(samples):

                    # Style points
                    progress_bar(
                        f"N={n_particles} \t Time={max_time} \t sample={sample_idx+1}",
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
                        n_particles,
                        n_iterations=None,
                        max_time=max_time,
                    )
        else:
            for j, n_iterations in enumerate(n_iterations_list):
                for sample_idx in range(samples):

                    # Style points
                    progress_bar(
                        f"N={n_particles} \t iterations={n_iterations} \t sample={sample_idx+1}",
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
                        n_particles,
                        n_iterations=n_iterations,
                        max_time=None,
                    )
    # Plotting
    plt.figure()
    for i, n_particles in enumerate(n_particles_list):
        x = np.array(n_iterations_list or times_list)
        y_mean = results[:, :, i].mean(axis=0)
        plt.plot(x, y_mean, "-o", label=f"N={n_particles}") # Plot mean return time
    
    # Theoretical values line
    plt.axhline(
        y=expected_return_time,
        color="r",
        linestyle="--",
        label=f"Theoretical Value ({expected_return_time:.4f})",
    )

    # Labels and saving
    if times_list is not None:
        plt.xlabel("Time")
    else:
        plt.xlabel("Number of iterations")
    plt.ylabel("Estimated Average Return Time")
    suffix = "iterations" if n_iterations_list is not None else "time units"
    plt.title(f"Estimated Return Time Fixing {suffix}")
    plt.legend()
    plt.savefig(plots_path / f"Ex2_return_time_{suffix.replace(" ", "_")}_comp.png")


def main():
    set_seed(42)
    G, LAMBDA, nodes, omega, P_bar, pi_bar = create_ex2_graph(plot_dir=Path(__file__).parent / "plots" / "Ex2_graph.png"
    )

    # Create values needed for simulation
    starting_node = "a"
    omega_star = omega.max()
    mapping = {node: idx for idx, node in enumerate(nodes)}
    expected_return_time = 1 / (
        omega[mapping[starting_node]] * pi_bar[mapping[starting_node]]
    )
    print(
        f"Theoretical expected return time to node '{starting_node}': {expected_return_time}"
    )

    # Run simulation over 20 samples, changing number of particles and iterations/time
    run_particle_simulation(
        starting_node,
        mapping,
        P_bar,
        omega_star,
        expected_return_time,
        samples=20,
        n_particles_list=[1, 10, 100],
        n_iterations_list=[100, 500, 1000, 5000, 10000],
        times_list=None,
        nodes=nodes,
        plots_path=Path(__file__).parent / "plots",
    )

    # Run simulation over 20 samples, changing number of particles and time limits
    run_particle_simulation(
        starting_node,
        mapping,
        P_bar,
        omega_star,
        expected_return_time,
        samples=20,
        n_particles_list=[1, 10, 100],
        n_iterations_list=None,
        times_list = list(range(10, 200, 10)),
        nodes=nodes,
        plots_path=Path(__file__).parent / "plots",
    )


if __name__ == "__main__":
    main()
