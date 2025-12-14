# Homework II — Network Dynamics and Learning (HW2)

This folder contains the code and material for **Homework II** (Network Dynamics and Learning, 2025).
The implementation follows the assignment requirements: **continuous-time random walks**, **French–DeGroot opinion dynamics**, and **multi-particle dynamics** on both **closed** and **open** networks.

## Project structure

* `Ex1.py` — **Problem 1**

  * Continuous-time random walk simulation on the **closed** network (matrix Λ in the assignment).
  * Empirical estimates of:

    * return time to node **a**
    * hitting time from **o → d**
  * Theoretical computations for return/hitting times (via linear systems).
  * French–DeGroot dynamics on the original graph and on the modified graphs required in points **(g)** and **(h)**.
  * Saves plots under `Homework_2/plots/ex1/`.

* `Ex2.py` — **Problem 2**

  * Multi-particle continuous-time dynamics (N=100) using:

    * **particle perspective** (system-wide Poisson clock)
    * **node perspective** (system-wide Poisson clock + node selection proportional to occupancy)
  * Return-time comparison (multi-particle vs single particle) and distribution/trajectory plots.
  * Saves plots under `Homework_2/plots/`.

* `Ex3.py` — **Problem 3**

  * Open network simulation with **input Poisson process** at node **o**.
  * Two scenarios:

    * **proportional rates** (rate ∝ number of particles in node)
    * **fixed rates** (node fires at fixed rate when non-empty)
  * Produces stacked-area plots of particle counts over time and prints summary statistics.
  * Saves plots under `Homework_2/plots/`.

* `utils.py` — helper utilities used across exercises (graph creation, seeding, DeGroot simulator, plot saving, etc.).

## How to run

From the `Homework_2/` directory:

```bash
python Ex1.py
python Ex2.py
python Ex3.py
```

All figures are saved automatically in the corresponding `plots/` subfolders.

## Dependencies

Main dependencies used:

* `numpy`
* `networkx`
* `matplotlib`
* `scipy` (used in Ex1 for Gaussian PDF comparison)

Install (example):

```bash
pip install numpy networkx matplotlib scipy
```

---

**Report:**

The final PDF report is located at each related homework's root [Nocita_Report_HW2.pdf](./Nocita_Report_HW2.pdf)

It contains:

* Explanation of methods
* All plotted results
* Answers to theoretical questions

**Collaboration**:
S346205 Luciano Scarpino, S334015 Andrea Vasco Grieco, S329057 shadi mahboubpardahi, S346378 Salvatore Nocita
