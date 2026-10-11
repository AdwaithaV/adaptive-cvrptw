"""
compare_homberger_500.py

Controlled fixed-vs-adaptive comparison on 500
Homberger CVRPTW customers.
"""

from pathlib import Path

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


def print_results(
    name,
    graph,
    coarsener,
    quality,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Final nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Final edges:",
        graph.number_of_edges(),
    )

    print(
        "Runtime:",
        coarsener.runtime_seconds,
    )

    print(
        "Candidate checks:",
        coarsener.total_candidate_checks,
    )

    print(
        "Infeasible checks:",
        coarsener.total_infeasible_checks,
    )

    print(
        "Density:",
        nx.density(graph),
    )

    print(
        "Average degree:",
        sum(
            dict(
                graph.degree()
            ).values()
        )
        / graph.number_of_nodes(),
    )

    print(
        "Clustering:",
        nx.average_clustering(
            graph
        ),
    )

    print(
        "Mean spatial compactness:",
        quality[
            "mean_spatial_compactness"
        ],
    )

    print(
        "Median spatial compactness:",
        quality[
            "median_spatial_compactness"
        ],
    )

    print(
        "Mean temporal dispersion:",
        quality[
            "mean_temporal_dispersion"
        ],
    )

    print(
        "Median temporal dispersion:",
        quality[
            "median_temporal_dispersion"
        ],
    )

    print(
        "Mean capacity utilization:",
        quality[
            "mean_capacity_utilization"
        ],
    )

    print(
        "Max capacity utilization:",
        quality[
            "max_capacity_utilization"
        ],
    )

    print(
        "Sequence feasibility rate:",
        quality[
            "sequence_feasibility_rate"
        ],
    )

    print(
        "Feasible clusters:",
        quality[
            "sequence_feasible_clusters"
        ],
        "/",
        quality[
            "num_clusters"
        ],
    )

    print(
        "Largest cluster:",
        quality[
            "largest_cluster"
        ],
    )


def main():

    dataset_path = Path(
        "data/datasets/homberger/C1_10_1.TXT"
    )

    instance = (
        load_homberger_instance(
            dataset_path
        )
    )

    customers = (
        instance["customers"]
        .head(500)
        .copy()
    )

    graph = build_graph(
        customers=customers,
        k=5,
        exclude_depot=True,
    )

    capacity = float(
        instance[
            "vehicle_capacity"
        ]
    )

    print("=" * 70)
    print(
        "500-CUSTOMER FIXED VS ADAPTIVE"
    )
    print("=" * 70)

    print(
        "Original nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Original edges:",
        graph.number_of_edges(),
    )

    #
    # FIXED
    #

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

    fixed_quality = (
        ClusterQualityEvaluator(
            fixed_graph,
            capacity,
        ).evaluate()
    )

    #
    # ADAPTIVE
    #

    adaptive = AdaptiveGraphCoarsener(
        graph=graph,
        vehicle_capacity=capacity,
        require_sequence_feasibility=True,
    )

    adaptive_graph = (
        adaptive.run(
            target_ratio=0.50,
            verbose=False,
        )
    )

    adaptive_quality = (
        ClusterQualityEvaluator(
            adaptive_graph,
            capacity,
        ).evaluate()
    )

    print_results(
        "FIXED RESULTS",
        fixed_graph,
        fixed,
        fixed_quality,
    )

    print_results(
        "ADAPTIVE RESULTS",
        adaptive_graph,
        adaptive,
        adaptive_quality,
    )

    print()
    print("=" * 70)
    print("ADAPTIVE - FIXED DIFFERENCE")
    print("=" * 70)

    print(
        "Spatial compactness difference:",
        adaptive_quality[
            "mean_spatial_compactness"
        ]
        -
        fixed_quality[
            "mean_spatial_compactness"
        ],
    )

    print(
        "Temporal dispersion difference:",
        adaptive_quality[
            "mean_temporal_dispersion"
        ]
        -
        fixed_quality[
            "mean_temporal_dispersion"
        ],
    )

    print(
        "Sequence feasibility difference:",
        adaptive_quality[
            "sequence_feasibility_rate"
        ]
        -
        fixed_quality[
            "sequence_feasibility_rate"
        ],
    )

    print(
        "Capacity utilization difference:",
        adaptive_quality[
            "mean_capacity_utilization"
        ]
        -
        fixed_quality[
            "mean_capacity_utilization"
        ],
    )


if __name__ == "__main__":
    main()
