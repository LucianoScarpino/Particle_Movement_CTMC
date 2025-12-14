from pathlib import Path
from matplotlib import pyplot as plt
from utils import create_ex2_graph, set_seed
import numpy as np
from typing import List, Dict, Optional, Tuple


def simulate_particle_return_time(
    start_node: str,
    P_bar: np.ndarray,
    nodes: List[str],
    mapping: Dict[str, int],
    rate: float,
    n_particles: int,
    until_return: bool = False,
    n_iterations: Optional[int] = None,
    max_time: Optional[float] = None,
) -> float:
    """Simulates particles moving on the graph until they return to the start node."""
    # Initialize remaining nodes count
    remaining_nodes = n_particles

    # Set initial positions and counters/trackers
    positions = np.array([start_node] * n_particles)
    total_time = 0.0
    return_counts = np.zeros(n_particles)
    return_total_times = np.zeros(n_particles)  # Total return times for each particle
    has_returned = np.zeros(n_particles, dtype=bool)  # Flag for each particle

    if until_return:
        while not np.all(has_returned):  # iterate until all particles have returned
            # Get next transition time and chosen particle
            t_next = -np.log(np.random.rand()) / rate
            chosen_particle = np.random.randint(0, remaining_nodes)

            # Move the chosen particle
            remaining_positions = positions[~has_returned]
            current_node = remaining_positions[chosen_particle]
            next_pos = np.random.choice(nodes, p=P_bar[mapping[current_node], :])

            # Update positions and total time
            positions_indices = np.where(~has_returned)[0]
            particle_idx = positions_indices[chosen_particle]
            positions[particle_idx] = next_pos
            total_time += t_next

            # Check for return to start node and remove particle if returned
            if next_pos == start_node and current_node != start_node:
                return_counts[particle_idx] += 1
                return_total_times[particle_idx] = total_time
                has_returned[particle_idx] = True
                rate = rate / remaining_nodes * (remaining_nodes - 1)
                remaining_nodes -= 1
    else:
        # Safety check
        assert n_iterations is not None or max_time is not None
        it = 0
        finished = False
        while not finished:
            # Get next transition time and chosen particle
            t_next = -np.log(np.random.rand()) / rate
            chosen_particle = np.random.randint(0, n_particles)

            # Move the chosen particle
            current_node = positions[chosen_particle]
            next_pos = np.random.choice(nodes, p=P_bar[mapping[current_node], :])
            positions[chosen_particle] = next_pos

            # Update total time
            total_time += t_next
            it += 1

            # Check for return to start node and update counters
            if next_pos == start_node and current_node != start_node:
                return_counts[chosen_particle] += 1
                return_total_times[chosen_particle] = total_time

            # Check termination conditions
            finished = (n_iterations is not None and it >= n_iterations) or (
                max_time is not None and total_time >= max_time
            )

    # Compute average return time and return
    average_return_time = np.sum(return_total_times) / np.sum(return_counts)
    return average_return_time


def simulate_node_distributions(
    start_node: str,
    P_bar: np.ndarray,
    nodes: List[str],
    mapping: dict,
    rate: float,
    n_particles: int,
    max_time: float,
):
    """Simulates particles moving on the graph and tracks node visit distributions."""
    # Set initial positions and counters
    current_distribution = np.zeros((len(nodes),))
    current_distribution[mapping[start_node]] = n_particles
    total_time = 0.0
    history = [current_distribution.copy()]
    time_steps = [0]

    # Run simulation until max_time is reached
    while total_time < max_time:
        # Get next transition time and chosen particle
        t_next = -np.log(np.random.rand()) / rate
        p = current_distribution / np.sum(current_distribution)

        # Choose a node based on current distribution and move a particle
        node_chosen = np.random.choice(nodes, p=p)
        next_pos = np.random.choice(nodes, p=P_bar[mapping[node_chosen], :])
        current_distribution[mapping[node_chosen]] -= 1
        current_distribution[mapping[next_pos]] += 1

        # Update total time, record history and time steps
        total_time += t_next
        history.append(current_distribution.copy())
        time_steps.append(total_time)

    # Compute visit distribution and return
    visit_distribution = current_distribution / np.sum(current_distribution)
    final_distribution = current_distribution.copy()
    return visit_distribution, history, time_steps, final_distribution


def plot_history(
    history: List[np.ndarray],
    nodes: List[str],
    time_steps: List[float],
    ground_truth: np.ndarray,
    plot_path: Path,
) -> None:
    """Generates a stacked area plot showing the distribution of particles over nodes over time."""
    # fmt: off
    # Define colors for each node
    colors = {"o": "tab:blue", "a": "tab:orange", "b": "tab:green", "c": "tab:red", "d": "tab:purple"}
    
    # Create stacked area plot
    plt.stackplot(range(len(history)), np.array(history).T, labels=nodes, alpha=0.8,
                  colors=[colors[node] for node in nodes])
    
    # Customize x-ticks to show time steps
    plt.xticks(len(history) // 10 * np.arange(11), labels=[
            f"{time_steps[i]:.0f}" for i in (len(time_steps) - 1) // 10 * np.arange(11)
        ],
    )
    
    # Alignment values for right bar
    previous_value = 0
    ground_truth_position = len(history) + 250
    
    # Add label to the right of the bar
    plt.text(ground_truth_position + 100, previous_value + 10, 
             r"Ground Truth ($\bar\pi$)", rotation=90)

    # Draw the ground truth bar
    for i, node in enumerate(nodes):
        plt.vlines(ground_truth_position, previous_value, previous_value + ground_truth[i] * 100, 
            colors=colors[node], linewidth=5,
        )
        previous_value += ground_truth[i] * 100

    # Finalize and save plot
    plt.legend(ncol=len(nodes), bbox_to_anchor=(0.5, 0.95), loc="upper center", framealpha=1)
    plt.xlabel("Time Units")
    plt.ylabel("Number of Particles")
    plt.title("Node Visit Distribution Over Time")
    plt.savefig(plot_path)
    plt.clf()
    # fmt: on


def barplot_final_results(
    mean_history,
    ci,
    ground_truth,
    nodes,
    plot_path: Path,
) -> None:
    """Generates a bar plot comparing the simulated and theoretical node visit distributions."""
    x = np.arange(len(nodes))
    plt.bar(x, ground_truth, label="Theoretical", width=0.9, alpha=0.7)
    plt.bar(x, mean_history, yerr=ci, label="Simulated", width=0.9, alpha=0.7)
    plt.xlabel("Nodes")
    plt.ylabel("Proportion of Visits")
    plt.title("Node Visit Distribution: Simulated vs Theoretical")
    plt.xticks(x, nodes)
    plt.legend()
    plt.savefig(plot_path)
    plt.clf()


def particle_simulation_multiple_runs(
    start_node: str,
    mapping: Dict[str, int],
    P_bar: np.ndarray,
    omega_star: float,
    expected_return_time: float,
    samples: int,
    nodes: List[str],
    n_particles: int,
    single_particle_iterations: int,
    plots_path: Path,
) -> Tuple[float, float, float, float]:
    """Compares average return times using multiple particles vs single particle simulations."""
    # Calculate rate for multiple particles
    rate = omega_star * n_particles
    average_return_times = []

    for _ in range(samples):
        # Simulate return time with multiple particles
        average_return_time = simulate_particle_return_time(
            start_node=start_node,
            P_bar=P_bar,
            nodes=nodes,
            mapping=mapping,
            rate=rate,
            n_particles=n_particles,
            n_iterations=None,
            until_return=True,
        )
        average_return_times.append(average_return_time)

    # Compute average return time and standard deviation
    avg_return_time = np.mean(average_return_times)
    std = np.std(average_return_times, ddof=1)

    # Simulate return time with single particle
    average_return_times_one_particle = []
    for _ in range(samples):
        average_return_time = simulate_particle_return_time(
            start_node=start_node,
            P_bar=P_bar,
            nodes=nodes,
            mapping=mapping,
            rate=omega_star,
            n_particles=1,
            n_iterations=single_particle_iterations,
            until_return=False,
        )
        average_return_times_one_particle.append(average_return_time)

    # Compute average return time and standard deviation for single particle
    avg_return_time_one_particle = np.mean(average_return_times_one_particle)
    std_one_particle = np.std(average_return_times_one_particle, ddof=1)

    # fmt: off
    print(f"Average Return Time over {samples} runs with multiple particles: {avg_return_time} +- {std}")
    print(f"Average Return Time over {samples} runs with single particle: {avg_return_time_one_particle} +- {std_one_particle}")
    
    # Generate comparison bar plot
    plt.figure()
    
    # Bar plot comparing multiple particles vs single particle
    plt.bar(["Multiple Particles", f"Single Particle ({single_particle_iterations} iterations)"],
        [avg_return_time, avg_return_time_one_particle],
        color=["tab:blue", "tab:orange"],
        yerr=[std, std_one_particle]
    )
    
    # Horizontal line for theoretical expected return time with label
    plt.axhline(y=expected_return_time, color="r", linestyle="--")
    plt.text(0.2, expected_return_time + 0.1, f"Theoretical Value ({expected_return_time:.2f})", color="r")
    
    # Annotation for error bars
    plt.text(1.53, 0.2, "*Error bars represent 1 standard deviation over samples", ha="center",
        fontsize=8, rotation=90,
    )
    
    # Finalize and save plot
    plt.ylabel("Average Return Time")
    plt.title(f"Multiple vs Single Particle, averaged over {samples} runs")
    plt.savefig(plots_path / "Ex2_return_time_comparison.png")
    plt.clf()
    # fmt: on

    # Return computed values
    return avg_return_time, std, avg_return_time_one_particle, std_one_particle


def plot_single_particle_standard_deviation(
    n_iterations: List[int],
    starting_node: str,
    mapping: Dict[str, int],
    P_bar: np.ndarray,
    omega_star: float,
    samples: int,
    nodes: List[str],
    plots_path: Path,
    compare_to_std: float,
):
    """Plot the standard deviation of the average return time for a single particle over different numbers of iterations."""
    # Initialize list to store standard deviations
    stds = []
    # Iterate over different numbers of iterations
    for n_iter in n_iterations:

        # Simulate multiple runs for current number of iterations and collect average return times
        average_return_times = []
        for _ in range(samples):
            average_return_time = simulate_particle_return_time(
                start_node=starting_node,
                P_bar=P_bar,
                nodes=nodes,
                mapping=mapping,
                rate=omega_star,
                n_particles=1,
                n_iterations=n_iter,
                until_return=False,
            )
            average_return_times.append(average_return_time)
        # Compute standard deviation for current number of iterations and store
        stds.append(np.std(average_return_times, ddof=1))

    # Plotting
    plt.figure()
    label_single = "Empirical std deviation (single particle)"
    label_multiple = f"Std from multiple particles simulation ({compare_to_std:.2f})"
    plt.plot(range(len(stds)), stds, marker="s", label=label_single)
    plt.axhline(y=compare_to_std, color="r", linestyle="--", label=label_multiple)

    # Finalize and save plot
    plt.legend()
    plt.xticks(range(len(n_iterations)), labels=n_iterations)
    plt.xlabel("Number of iterations for single particle")
    plt.ylabel("Average Return Time")
    plt.title(f"Single Particle std deviation, estimated from {samples} runs")
    plt.savefig(plots_path)
    plt.clf()


def simulate_node_distributions_multiple_runs(
    start_node: str,
    P_bar: np.ndarray,
    nodes: List[str],
    mapping: Dict[str, int],
    rate: float,
    n_particles: int,
    max_time: float,
    samples: int,
    pi_bar: np.ndarray,
    plots_path: Path,
):
    """Simulates node visit distributions over multiple runs and plots the averaged results with confidence intervals."""
    # Initialize array to store visit distributions for each run
    aggregated_visit_distribution = np.zeros((len(nodes), samples))

    # Run simulations and collect visit distributions
    for i in range(samples):
        _, _, _, final_distribution = simulate_node_distributions(
            start_node, P_bar, nodes, mapping, rate, n_particles, max_time
        )
        aggregated_visit_distribution[:, i] = final_distribution

    # Normalize visit distributions and compute statistics
    visit_distribution_normalized = aggregated_visit_distribution / n_particles
    averaged_visit_distribution_normalized = np.mean(
        visit_distribution_normalized, axis=1
    )
    std_visit_distribution_normalized = np.std(
        visit_distribution_normalized, axis=1, ddof=1
    )

    # Compute 95% confidence intervals for normalized visit distributions
    ci_95_normalized = 1.96 * std_visit_distribution_normalized / np.sqrt(samples)

    # fmt: off
    plt.bar(nodes, averaged_visit_distribution_normalized, yerr=ci_95_normalized,
        alpha=0.7, label="Simulated", width=0.9, align="center"
    )
    plt.bar(nodes, pi_bar, alpha=0.7, label="Theoretical", width=0.9, align="center")
    # fmt: on

    # Finalize and save plot
    plt.xlabel("Nodes")
    plt.ylabel("Proportion of Visits")
    plt.title(f"Node Visit Distribution: average over {samples} runs")
    plt.legend()
    plt.savefig(plots_path / "node_visit_distribution_multiple_runs.png")
    plt.clf()

    # Compute final averaged distribution and confidence intervals on raw counts
    final_distribution = aggregated_visit_distribution.mean(axis=1)
    std_final_distribution = aggregated_visit_distribution.std(axis=1, ddof=1)
    ci_95 = 1.96 * std_final_distribution / np.sqrt(samples)

    # Return results
    return final_distribution, ci_95


def main():
    set_seed(42)
    _, _, nodes, omega, P_bar, pi_bar = create_ex2_graph(
        plot_dir=Path(__file__).parent / "plots" / "Ex2" / "graph.png"
    )

    # -------------------------------------------------------------
    # ------------------------- Point (a) -------------------------
    # -------------------------------------------------------------

    # Compute theoretical expected return time
    starting_node = "a"
    omega_star = omega.max()
    expected_return_time = 1 / (
        omega[nodes.index(starting_node)] * pi_bar[nodes.index(starting_node)]
    )
    print(
        f"Theoretical expected return time to node '{starting_node}': {expected_return_time}"
    )

    # Map nodes to indices
    mapping = {node: idx for idx, node in enumerate(nodes)}
    n_particles = 100
    rate = omega_star * n_particles

    # Run particle simulation for return times
    _, std, _, _ = particle_simulation_multiple_runs(
        starting_node,
        mapping,
        P_bar,
        omega_star,
        expected_return_time,
        samples=100,
        nodes=nodes,
        n_particles=n_particles,
        single_particle_iterations=671,
        plots_path=Path(__file__).parent / "plots" / "Ex2",
    )
    # Plot standard deviation for single particle simulations
    plot_single_particle_standard_deviation(
        n_iterations=[300, 671, 1000],
        starting_node=starting_node,
        mapping=mapping,
        P_bar=P_bar,
        omega_star=omega_star,
        samples=100,
        nodes=nodes,
        plots_path=Path(__file__).parent
        / "plots"
        / "Ex2"
        / f"return_time_single_particle_std.png",
        compare_to_std=std,
    )

    # -------------------------------------------------------------
    # ------------------------- Point (b) -------------------------
    # -------------------------------------------------------------

    # Set parameters for node distribution simulation
    max_time = 60
    n_particles = 100

    # Run simulation for node visit distributions
    visit_distribution, history, time_steps, final_distribution = (
        simulate_node_distributions(
            start_node=starting_node,
            P_bar=P_bar,
            nodes=nodes,
            mapping=mapping,
            rate=rate,
            n_particles=n_particles,
            max_time=max_time,
        )
    )

    # Print final results
    print("Simulated final distribution over nodes:")
    for node in nodes:
        print(f"Node {node}: {final_distribution[mapping[node]]:.4f}")
    print("Ground truth distribution (pi_bar*N):", pi_bar * n_particles)

    # Plot history of node visit distributions
    plot_path = Path(__file__).parent / "plots" / "Ex2" / "node_visit_distribution.png"
    plot_history(history, nodes, time_steps, ground_truth=pi_bar, plot_path=plot_path)

    # Compute and print mean and 95% CI for final visit distribution
    mean_history = np.array(history).mean(axis=0)
    std_history = np.array(history).std(axis=0, ddof=1)
    ci = 1.96 * std_history / np.sqrt(len(history))

    # Print final visit distribution with 95% confidence intervals
    print("Final visit distribution with 95% CI:")
    for node in nodes:
        print(
            f"Node {node}: {mean_history[mapping[node]]:.4f} +- {ci[mapping[node]]:.4f} (95% CI)"
        )

    # Normalize history and plot barplot of final results
    normalized_history = np.array(history) / n_particles
    mean_history_normalized = np.array(normalized_history).mean(axis=0)
    std_history_normalized = np.array(normalized_history).std(axis=0, ddof=1)
    ci_normalized = 1.96 * std_history_normalized / np.sqrt(len(normalized_history))
    barplot_final_results(
        mean_history_normalized,
        ci_normalized,
        pi_bar,
        nodes,
        Path(__file__).parent / "plots" / "Ex2" / "node_visit_barplot.png",
    )

    # Check for convergence to theoretical distribution over multiple runs
    averaged_visit_distribution, ci_95 = simulate_node_distributions_multiple_runs(
        start_node=starting_node,
        P_bar=P_bar,
        nodes=nodes,
        mapping=mapping,
        rate=rate,
        n_particles=n_particles,
        max_time=max_time,
        samples=1000,
        pi_bar=pi_bar,
        plots_path=Path(__file__).parent / "plots",
    )

    # Print averaged visit distribution with 95% confidence intervals
    print("Averaged visit distribution over multiple runs:")
    for node in nodes:
        print(
            f"Node {node}: {averaged_visit_distribution[mapping[node]]:.4f} +- {ci_95[mapping[node]]:.4f} (95% CI)"
        )


if __name__ == "__main__":
    main()
