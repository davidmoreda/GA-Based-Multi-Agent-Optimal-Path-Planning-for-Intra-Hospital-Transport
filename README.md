# GA-Based-Multi-Agent-Optimal-Path-Planning-for-Intra-Hospital-Transport

This repository contains the source code, data, and experimental results for the paper **"GA-Based Multi-Agent Optimal Path Planning for Intra-Hospital Transport"**.

The project proposes a comprehensive framework using **Evolutionary Computation (EC)** to generate safe, efficient, and collision-free routes for multiple autonomous agents performing Intra-Hospital Transport (IHT) missions (pickup and delivery) within a constrained hospital environment.

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| **Problem Formulation** | Modeling of the hospital environment as a directed graph, formalizing the task as a **Multi-Agent Optimal Path Planning (MAOPP)** problem with operational and critical safety constraints. |
| **Multi-Objective Optimization** | Use of **NSGA-II** to explicitly balance the trade-off between **geometric path distance** ($F_{\text{clean}}$) and **safety compliance** ($F_{\text{pen}}$), which includes penalties for collisions and insufficient inter-agent separation. |
| **Algorithmic Comparison** | Implementation and systematic grid-search evaluation of four metaheuristic optimizers over a shared route-encoding framework: **Genetic Algorithm (GA)**, **Simulated Annealing (SA)**, **$(\mu+\lambda)$ Evolution Strategy (ES)**, and **NSGA-II**. |
| **Real-World Environment** | Simulation and validation performed on a graph derived from a real-world hospital floor plan using image segmentation. |

---

## 💻 Project Structure

The repository is organized into several key directories:

.
├── algorithms/                 # Core implementation of the evolutionary algorithms and MAOPP logic
│   ├── ga_core.py              # Encoding, evaluation, genetic operators (mutation, crossover)
│   ├── ga_runner.py            # Single-objective GA execution script
│   ├── mulambda_runner.py      # (μ+λ) Evolution Strategy execution script
│   ├── sa_runner.py            # Simulated Annealing execution script
│   └── ga_runner_multi.py      # NSGA-II multi-objective execution script
├── data/                       # Input files for the environment and missions
│   └── Floorplan/              # Hospital floorplan image files used for graph generation
├── hyperparametrization/       # Scripts used for the systematic grid-search evaluation
│   ├── grid_search.py          # Script for single-objective grid search (GA, ES, SA)
│   └── grid_search_algos.py    # Script for multi-objective grid search (NSGA-II)
├── results/                    # Database files containing the complete grid-search results
├── figures/                    # Generated plots and visualization from the experiments
└── README.md

---

## ⚙️ Methodology and Results

### Single-Objective Performance

The systematic evaluation demonstrated the superiority of the **$(\mu+\lambda)$ Evolution Strategy** for minimizing the penalized distance $F_{\text{pen}}$.

| Algorithm | Min Penalized Distance | Mean Penalized Distance |
| :--- | :--- | :--- |
| **$(\mu+\lambda)$ ES** | **1104.66** | 1446.59 |
| GA | 1184.63 | 1453.53 |
| SA | 1989.32 | 2297.90 |

**Distribution of Penalized Distance**

A boxplot comparison of the single-objective performance:

![Distribution of Penalized Distance for GA, SA, and (μ+λ) ES](figures/fig_single_boxplots.png)

### Multi-Objective Optimization (NSGA-II)

The NSGA-II was used to find the Pareto-optimal front for the bi-objective problem: $\min(F_{\text{clean}}, F_{\text{pen}})$. The best configuration achieved a hypervolume exceeding **$1.54 \times 10^7$**, generating high-quality solutions that balance efficiency and safety.

**Best Achieved Pareto Front**

This plot illustrates the fundamental trade-off, where increasing safety (reducing $F_{\text{pen}}$) often requires a slightly longer path (increasing $F_{\text{clean}}$).

![Best Pareto front identified by NSGA-II for the Clean vs. Penalized distance objectives](figures/fig_nsga2_pareto.png)

**Optimized Multi-Agent Routes**

The framework successfully generates safe and feasible trajectories on the real hospital map. The visualization below shows different solutions from the Pareto front, demonstrating how timing and detours are adjusted to meet the required safety separation $\delta_{\min}$.

![Comparison of three representative NSGA-II solutions (Best Clean, Best Tradeoff, and Best Penalized)](figures/Routes_multiobjective_comparison_10_1000_06_04_s42.png)

---

## 📄 Paper Reference

If you use this code or methodology in your research, please cite the corresponding paper:

**Title:** GA-Based Multi-Agent Optimal Path Planning for Intra-Hospital Transport
**Authors:** David Moreda Amezcua and Carmen Gutiérrez Silva
**Journal:** *IEEE Access*
**DOI:** 10.1109/ACCESS.2017.DOI (Placeholder)
**Date:** December 30, 2025 (Placeholder)

@article{amezcua2025ga,
  title={GA-Based Multi-Agent Optimal Path Planning for Intra-Hospital Transport},
  author={Amezcua, David Moreda and Silva, Carmen Guti{'e}rrez},
  journal={IEEE Access},
  year={2025},
  volume={XX},
  number={XX},
  pages={XXXX-XXXX},
  doi={10.1109/ACCESS.2017.DOI}
}
---

## 🖋️ Authors and Contributors

* **David Moreda Amezcua** (dmoredaamezcua@al.uloyola.es)
* **Carmen Gutiérrez Silva** (cgutierrezsilva@al.uloyola.es)

Loyola University Andalusia, Department of AI, Seville, Spain.
