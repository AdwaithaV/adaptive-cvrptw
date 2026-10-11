"""
benchmark_reduction_ratios.py

Final Phase-1 robustness experiment.

Tests Fixed and Adaptive-v2 graph coarsening on
representative Homberger CVRPTW families across
multiple reduction ratios.

Families:
    C1
    R1
    RC1

Targets:
    1000 -> 750
    1000 -> 500
    1000 -> 250
"""

from pathlib import Path
import csv

import networkx as nx

from algorithms.adaptive_coarsening import (
    AdaptiveGraphCoarsener,
)
from algorithms.fixed_coarsening import (
    FixedGraphCoarsener,
)
from data.homberger_loader import (
    load_homberger_instance,
)
from graph.build_graph import build_graph
from metrics.cluster_quality import (
    ClusterQualityEvaluator,
)


DATASET_DIR = Path(
    "data/datasets/homberger"
)

INSTANCES = [
    "C1_10_1.TXT",
    "R1_10_1.TXT",
    "RC1_10_1.TXT",
]

TARGET_RATIOS = [
    0.75,
    0.50,
    0.25,
]

K = 5

SEQUENCE_WEIGHT = 0.05

FIXED_ALPHA = 0.50
FIXED_BETA = 0.50
FIXED_GAMMA = 0.10


def evaluate(
    instance_name,
    family,
    method,
    target_ratio,
    original_graph,
    result_graph,
    coarsener,
    capacity,
):

    quality = (
        ClusterQualityEvaluator(
            result_graph,
            capacity,
        ).evaluate()
    )

    validation = (
        coarsener.validate()
    )

    original_nodes = (
        original_graph.number_of_nodes()
    )

    final_nodes = (
        result_graph.number_of_nodes()
    )

    reduction_fraction = (
        1.0
        - (
            final_nodes
            / original_nodes
        )
    )

    average_degree = (
        sum(
            dict(
                result_graph.degree()
            ).values()
        )
        / final_nodes
    )

    return {
        "instance":
            instance_name,

        "family":
            family,

        "method":
            method,

        "target_ratio":
            target_ratio,

        "reduction_fraction":
            reduction_fraction,

        "original_nodes":
            original_nodes,

        "final_nodes":
            final_nodes,

        "successful_contractions":
            coarsener.total_successful_contractions,

        "final_edges":
            result_graph.number_of_edges(),

        "density":
            nx.density(
                result_graph
            ),

        "average_degree":
            average_degree,

        "clustering":
            nx.average_clustering(
                result_graph
            ),

        "runtime_seconds":
            coarsener.runtime_seconds,

        "candidate_checks":
            coarsener.total_candidate_checks,

        "infeasible_checks":
            coarsener.total_infeasible_checks,

        "spatial_compactness":
            quality[
                "mean_spatial_compactness"
            ],

        "median_spatial_compactness":
            quality[
                "median_spatial_compactness"
            ],

        "temporal_dispersion":
            quality[
                "mean_temporal_dispersion"
            ],

        "median_temporal_dispersion":
            quality[
                "median_temporal_dispersion"
            ],

        "sequence_feasibility_rate":
            quality[
                "sequence_feasibility_rate"
            ],

        "mean_capacity_utilization":
            quality[
                "mean_capacity_utilization"
            ],

        "max_capacity_utilization":
            quality[
                "max_capacity_utilization"
            ],

        "largest_cluster":
            quality[
                "largest_cluster"
            ],

        "average_cluster_size":
            quality[
                "average_cluster_size"
            ],

        "connected":
            validation[
                "connected"
            ],

        "membership_preserved":
            validation[
                "membership_preserved"
            ],

        "duplicate_members":
            validation[
                "duplicate_members"
            ],

        "demand_preserved":
            validation[
                "demand_preserved"
            ],

        "capacity_valid":
            validation[
                "capacity_valid"
            ],
    }


def print_result(result):

    print(
        f"{result['family']:4s} | "
        f"{result['method']:10s} | "
        f"{result['original_nodes']:4d}"
        f"->{result['final_nodes']:4d} | "
        f"Spatial="
        f"{result['spatial_compactness']:9.3f} | "
        f"Temporal="
        f"{result['temporal_dispersion']:9.3f} | "
        f"Seq="
        f"{result['sequence_feasibility_rate']:.3f} | "
        f"Runtime="
        f"{result['runtime_seconds']:8.2f}s"
    )


def run_method(
    instance_name,
    family,
    graph,
    capacity,
    target_ratio,
    method,
):

    if method == "FIXED":

        coarsener = FixedGraphCoarsener(
            graph=graph,
            vehicle_capacity=capacity,
            require_sequence_feasibility=True,
            alpha=FIXED_ALPHA,
            beta=FIXED_BETA,
            gamma=FIXED_GAMMA,
        )

    elif method == "ADAPTIVE":

        coarsener = AdaptiveGraphCoarsener(
            graph=graph,
            vehicle_capacity=capacity,
            require_sequence_feasibility=True,
            sequence_weight=SEQUENCE_WEIGHT,
        )

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    result_graph = coarsener.run(
        target_ratio=target_ratio,
        verbose=False,
    )

    result = evaluate(
        instance_name=instance_name,
        family=family,
        method=method,
        target_ratio=target_ratio,
        original_graph=graph,
        result_graph=result_graph,
        coarsener=coarsener,
        capacity=capacity,
    )

    print_result(
        result
    )

    return result


def main():

    print("=" * 110)
    print(
        "HOMBERGER REDUCTION-RATIO ROBUSTNESS BENCHMARK"
    )
    print("=" * 110)

    print(
        "Families:",
        ", ".join(INSTANCES)
    )

    print(
        "Target ratios:",
        TARGET_RATIOS
    )

    print(
        "Adaptive sequence weight:",
        SEQUENCE_WEIGHT
    )

    print()

    all_results = []

    for filename in INSTANCES:

        path = (
            DATASET_DIR
            / filename
        )

        if not path.exists():

            print(
                f"SKIPPING {filename}: "
                f"file not found."
            )

            continue

        print()
        print("=" * 110)
        print(
            "INSTANCE:",
            filename
        )
        print("=" * 110)

        instance = (
            load_homberger_instance(
                path
            )
        )

        customers = (
            instance[
                "customers"
            ]
            .head(1000)
            .copy()
        )

        capacity = float(
            instance[
                "vehicle_capacity"
            ]
        )

        graph = build_graph(
            customers=customers,
            k=K,
            exclude_depot=True,
        )

        family = (
            filename
            .split("_")[0]
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
            nx.is_connected(
                graph
            )
        )

        for target_ratio in TARGET_RATIOS:

            target_nodes = int(
                round(
                    graph.number_of_nodes()
                    * target_ratio
                )
            )

            print()
            print(
                "-" * 110
            )

            print(
                f"TARGET: "
                f"{graph.number_of_nodes()} "
                f"-> {target_nodes}"
            )

            print(
                "-" * 110
            )

            fixed_result = (
                run_method(
                    instance_name=filename,
                    family=family,
                    graph=graph,
                    capacity=capacity,
                    target_ratio=target_ratio,
                    method="FIXED",
                )
            )

            adaptive_result = (
                run_method(
                    instance_name=filename,
                    family=family,
                    graph=graph,
                    capacity=capacity,
                    target_ratio=target_ratio,
                    method="ADAPTIVE",
                )
            )

            all_results.append(
                fixed_result
            )

            all_results.append(
                adaptive_result
            )

            print()

            spatial_improvement = (
                fixed_result[
                    "spatial_compactness"
                ]
                -
                adaptive_result[
                    "spatial_compactness"
                ]
            )

            temporal_improvement = (
                fixed_result[
                    "temporal_dispersion"
                ]
                -
                adaptive_result[
                    "temporal_dispersion"
                ]
            )

            sequence_difference = (
                adaptive_result[
                    "sequence_feasibility_rate"
                ]
                -
                fixed_result[
                    "sequence_feasibility_rate"
                ]
            )

            print(
                "Adaptive spatial improvement:",
                spatial_improvement
            )

            print(
                "Adaptive temporal improvement:",
                temporal_improvement
            )

            print(
                "Adaptive sequence difference:",
                sequence_difference
            )

    if not all_results:

        print(
            "No experiments completed."
        )

        return

    output_file = (
        "homberger_reduction_ratio_benchmark.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print()
    print("=" * 110)
    print(
        "REDUCTION-RATIO BENCHMARK COMPLETE"
    )
    print("=" * 110)

    print(
        "Total experimental runs:",
        len(all_results)
    )

    print(
        "Results saved to:",
        output_file
    )


if __name__ == "__main__":
    main()
