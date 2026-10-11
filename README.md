# Adaptive Graph Coarsening for CVRPTW

Adaptive graph coarsening for the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**, developed during **QWorld QIntern 2026** under the mentorship of **Dr. Paweł Góra**.

This project investigates how routing graphs can be reduced while preserving information relevant to customer locations, time windows, and feasible visit sequences. It compares fixed and adaptive coarsening approaches using synthetic data and Homberger benchmark instances.

## Motivation

CVRPTW involves planning vehicle routes that serve customers while respecting vehicle capacities and customer time windows. As the number of customers grows, the routing problem becomes more expensive to solve.

Graph coarsening groups customers into clusters and contracts the original graph into a smaller representation. The challenge is to reduce graph size while retaining useful routing structure.

This repository explores that preprocessing step within the broader research project **“Solving routing problems using quantum annealing.”** The experiments here focus on graph construction, coarsening, and structural evaluation.

## Approach

The implementation includes:

- **Graph construction:** Represent routing instances as customer graphs.
- **Customer similarity:** Evaluate relationships between customers to guide clustering.
- **Fixed coarsening:** Provide a baseline for comparison.
- **Adaptive coarsening:** Use adaptive weights and sequence-aware similarity to guide graph reduction.
- **Graph contraction:** Build a reduced graph from customer clusters.
- **Evaluation:** Compare structural and cluster-quality metrics across methods and configurations.

The benchmark scripts examine different instance families, graph reduction ratios, and sequence-weight settings.

## Repository Structure

| Path | Purpose |
|---|---|
| `algorithms/` | Similarity, adaptive weights, fixed and adaptive coarsening, and contraction |
| `data/` | Synthetic data generation and dataset loaders |
| `data/datasets/homberger/` | Homberger benchmark instance files |
| `graph/` | Graph construction and graph utilities |
| `metrics/` | Evaluation and cluster-quality calculations |
| `visualization/` | Graph plotting and comparison utilities |
| `results/` | Benchmark summaries and generated figures |
| `main.py` | Main project entry point |
| `experiment.py` | Experiment script |
| `benchmark_all_families.py` | Benchmark comparisons across instance families |
| `benchmark_reduction_ratios.py` | Experiments with different reduction ratios |
| `sweep_sequence_weight_500.py` | Sequence-weight experiments |
| `test_homberger_*.py` | Scripts for checking individual pipeline components |

## Getting Started

### Installation

Use Python 3.11 to match the project’s development environment.

```bash
git clone https://github.com/AdwaithaV/adaptive-cvrptw.git
cd adaptive-cvrptw
python -m venv .venv
```

Activate the environment on Windows Command Prompt:

```bat
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the Main Pipeline

```bash
python main.py
```

### Run Experiments

```bash
python experiment.py
python benchmark_all_families.py
python benchmark_reduction_ratios.py
python sweep_sequence_weight_500.py
```

Before running benchmark scripts, review their dataset paths and experiment settings. Larger instances and parameter sweeps may take longer to complete.

## Datasets

The repository includes synthetic routing data and Homberger CVRPTW benchmark instances.

The included Homberger filenames cover the `C1`, `C2`, `R1`, `R2`, `RC1`, and `RC2` families, representing different customer distributions and time-window configurations.

Benchmark data remains attributable to its original creators. Inclusion in this repository does not imply ownership of the datasets.

## Evaluation and Results

The benchmark outputs examine measures such as:

- Spatial compactness
- Temporal dispersion
- Sequence feasibility
- Graph reduction ratio

Saved outputs include:

| File | Contents |
|---|---|
| `homberger_1000_results.csv` | Results from the 1,000-customer comparison |
| `homberger_family_benchmark.csv` | Benchmark results across instance families |
| `homberger_reduction_ratio_benchmark.csv` | Results across reduction ratios |
| `results/family_comparison_summary.csv` | Summary of family comparisons |
| `results/reduction_ratio_comparison_summary.csv` | Summary of reduction-ratio comparisons |

The `results/` folder also contains figures comparing spatial compactness, temporal dispersion, sequence feasibility, and changes across reduction ratios.

Interpret each result alongside its dataset, parameter settings, and metric definition. Coarsening quality can vary across instance families and reduction levels.

## Scope and Limitations

This repository evaluates graph reduction and the properties of the resulting clusters.

A smaller graph or an improvement in a cluster-quality metric does not by itself establish a better final vehicle route. Demonstrating routing benefits requires downstream route construction and evaluation of capacity constraints, time-window feasibility, travel cost, and total runtime.

The broader project is motivated by quantum annealing, but the files documented here do not establish quantum hardware execution or quantum advantage.

## Author and Acknowledgments

**Adwaitha V**  
B.Tech Electronics and Computer Engineering  
Amrita Vishwa Vidyapeetham, Amritapuri Campus

Developed during **QWorld QIntern 2026** under the mentorship of **Dr. Paweł Góra**.

Acknowledgments to QWorld, the project mentor, and the creators of the benchmark datasets.
