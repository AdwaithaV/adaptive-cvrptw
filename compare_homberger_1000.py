from pathlib import Path
import csv
import networkx as nx

from algorithms.adaptive_coarsening import AdaptiveGraphCoarsener
from algorithms.fixed_coarsening import FixedGraphCoarsener
from data.homberger_loader import load_homberger_instance
from graph.build_graph import build_graph
from metrics.cluster_quality import ClusterQualityEvaluator


def evaluate(name, graph, coarsener, capacity):

    quality = ClusterQualityEvaluator(
        graph,
        capacity
    ).evaluate()

    validation = coarsener.validate()

    avg_degree = (
        sum(dict(graph.degree()).values())
        / graph.number_of_nodes()
    )

    result = {
        "method": name,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "average_degree": avg_degree,
        "clustering": nx.average_clustering(graph),

        "runtime_seconds":
            coarsener.runtime_seconds,

        "candidate_checks":
            coarsener.total_candidate_checks,

        "infeasible_checks":
            coarsener.total_infeasible_checks,

        "spatial_compactness":
            quality["mean_spatial_compactness"],

        "temporal_dispersion":
            quality["mean_temporal_dispersion"],

        "sequence_feasibility_rate":
            quality["sequence_feasibility_rate"],

        "largest_cluster":
            quality["largest_cluster"],

        "average_cluster_size":
            quality["average_cluster_size"],

        "connected":
            validation["connected"],

        "membership_preserved":
            validation["membership_preserved"],

        "demand_preserved":
            validation["demand_preserved"],

        "capacity_valid":
            validation["capacity_valid"],
    }

    return result


def print_result(result):

    print()
    print("=" * 72)
    print(result["method"])
    print("=" * 72)

    for key, value in result.items():

        if key != "method":
            print(
                f"{key:30s}: {value}"
            )


def main():

    instance = load_homberger_instance(
        Path(
            "data/datasets/homberger/C1_10_1.TXT"
        )
    )

    customers = (
        instance["customers"]
        .head(1000)
        .copy()
    )

    capacity = float(
        instance["vehicle_capacity"]
    )

    print("=" * 72)
    print("HOMBERGER 1000 CUSTOMER EXPERIMENT")
    print("=" * 72)

    graph = build_graph(
        customers=customers,
        k=5,
        exclude_depot=True,
    )

    print(
        "Original nodes:",
        graph.number_of_nodes()
    )

    print(
        "Original edges:",
        graph.number_of_edges()
    )

    print(
        "Connected:",
        nx.is_connected(graph)
    )

    #
    # FIXED
    #

    print()
    print("Running FIXED 1000 -> 500...")

    fixed = FixedGraphCoarsener(
        graph=graph,
        vehicle_capacity=capacity,
        require_sequence_feasibility=True,
        alpha=0.5,
        beta=0.5,
        gamma=0.10,
    )

    fixed_graph = fixed.run(
        target_ratio=0.50,
        verbose=False,
    )

    fixed_result = evaluate(
        "FIXED",
        fixed_graph,
        fixed,
        capacity,
    )

    print_result(
        fixed_result
    )

    #
    # ADAPTIVE V2
    #

    print()
    print(
        "Running ADAPTIVE-V2 "
        "(sequence_weight=0.05)..."
    )

    adaptive = AdaptiveGraphCoarsener(
        graph=graph,
        vehicle_capacity=capacity,
        require_sequence_feasibility=True,
        sequence_weight=0.05,
    )

    adaptive_graph = adaptive.run(
        target_ratio=0.50,
        verbose=False,
    )

    adaptive_result = evaluate(
        "ADAPTIVE_V2_W005",
        adaptive_graph,
        adaptive,
        capacity,
    )

    print_result(
        adaptive_result
    )

    #
    # COMPARISON
    #

    print()
    print("=" * 72)
    print("ADAPTIVE V2 - FIXED")
    print("=" * 72)

    print(
        "Spatial difference:",
        adaptive_result[
            "spatial_compactness"
        ]
        -
        fixed_result[
            "spatial_compactness"
        ]
    )

    print(
        "Temporal difference:",
        adaptive_result[
            "temporal_dispersion"
        ]
        -
        fixed_result[
            "temporal_dispersion"
        ]
    )

    print(
        "Sequence feasibility difference:",
        adaptive_result[
            "sequence_feasibility_rate"
        ]
        -
        fixed_result[
            "sequence_feasibility_rate"
        ]
    )

    #
    # SAVE CSV
    #

    filename = "homberger_1000_results.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                list(
                    fixed_result.keys()
                ),
        )

        writer.writeheader()

        writer.writerow(
            fixed_result
        )

        writer.writerow(
            adaptive_result
        )

    print()
    print(
        "Results saved to:",
        filename
    )


if __name__ == "__main__":
    main()
