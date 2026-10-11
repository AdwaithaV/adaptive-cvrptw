from pathlib import Path
import csv
import os
import networkx as nx

from algorithms.adaptive_coarsening import AdaptiveGraphCoarsener
from algorithms.fixed_coarsening import FixedGraphCoarsener
from data.homberger_loader import load_homberger_instance
from graph.build_graph import build_graph
from metrics.cluster_quality import ClusterQualityEvaluator


DATASET_DIR = Path(
    "data/datasets/homberger"
)

INSTANCES = [
    "C1_10_1.TXT",
    "C2_10_1.TXT",
    "R1_10_1.TXT",
    "R2_10_1.TXT",
    "RC1_10_1.TXT",
    "RC2_10_1.TXT",
]

TARGET_RATIO = 0.50
SEQUENCE_WEIGHT = 0.05
K = 5


def evaluate(
    instance_name,
    family,
    method,
    graph,
    coarsener,
    capacity,
):

    quality = ClusterQualityEvaluator(
        graph,
        capacity,
    ).evaluate()

    validation = coarsener.validate()

    average_degree = (
        sum(
            dict(
                graph.degree()
            ).values()
        )
        / graph.number_of_nodes()
    )

    return {
        "instance":
            instance_name,

        "family":
            family,

        "method":
            method,

        "nodes":
            graph.number_of_nodes(),

        "edges":
            graph.number_of_edges(),

        "density":
            nx.density(graph),

        "average_degree":
            average_degree,

        "clustering":
            nx.average_clustering(
                graph
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

        "temporal_dispersion":
            quality[
                "mean_temporal_dispersion"
            ],

        "sequence_feasibility_rate":
            quality[
                "sequence_feasibility_rate"
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
        f"{result['method']:12s} | "
        f"Spatial="
        f"{result['spatial_compactness']:.4f} | "
        f"Temporal="
        f"{result['temporal_dispersion']:.4f} | "
        f"Seq="
        f"{result['sequence_feasibility_rate']:.4f} | "
        f"Runtime="
        f"{result['runtime_seconds']:.2f}s"
    )


def run_instance(filename):

    path = DATASET_DIR / filename

    if not path.exists():
        print(
            f"\nSKIPPING: {path} not found"
        )
        return []

    print()
    print("=" * 90)
    print(
        "INSTANCE:",
        filename
    )
    print("=" * 90)

    instance = (
        load_homberger_instance(
            path
        )
    )

    customers = (
        instance["customers"]
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

    print(
        "Original nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Original edges:",
        graph.number_of_edges(),
    )

    print(
        "Connected:",
        nx.is_connected(graph),
    )

    family = (
        filename
        .split("_")[0]
    )

    #
    # FIXED
    #

    print(
        "\nRunning FIXED..."
    )

    fixed = FixedGraphCoarsener(
        graph=graph,
        vehicle_capacity=capacity,
        require_sequence_feasibility=True,
        alpha=0.5,
        beta=0.5,
        gamma=0.10,
    )

    fixed_graph = fixed.run(
        target_ratio=TARGET_RATIO,
        verbose=False,
    )

    fixed_result = evaluate(
        filename,
        family,
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

    print(
        "Running ADAPTIVE..."
    )

    adaptive = AdaptiveGraphCoarsener(
        graph=graph,
        vehicle_capacity=capacity,
        require_sequence_feasibility=True,
        sequence_weight=SEQUENCE_WEIGHT,
    )

    adaptive_graph = adaptive.run(
        target_ratio=TARGET_RATIO,
        verbose=False,
    )

    adaptive_result = evaluate(
        filename,
        family,
        "ADAPTIVE",
        adaptive_graph,
        adaptive,
        capacity,
    )

    print_result(
        adaptive_result
    )

    print()
    print(
        "Spatial improvement:",
        fixed_result[
            "spatial_compactness"
        ]
        -
        adaptive_result[
            "spatial_compactness"
        ],
    )

    print(
        "Temporal improvement:",
        fixed_result[
            "temporal_dispersion"
        ]
        -
        adaptive_result[
            "temporal_dispersion"
        ],
    )

    print(
        "Sequence difference:",
        adaptive_result[
            "sequence_feasibility_rate"
        ]
        -
        fixed_result[
            "sequence_feasibility_rate"
        ],
    )

    return [
        fixed_result,
        adaptive_result,
    ]


def main():

    print("=" * 90)
    print(
        "HOMBERGER MULTI-FAMILY BENCHMARK"
    )
    print("=" * 90)

    print(
        "Reduction:",
        "1000 -> 500"
    )

    print(
        "Adaptive sequence weight:",
        SEQUENCE_WEIGHT
    )

    all_results = []

    for filename in INSTANCES:

        results = run_instance(
            filename
        )

        all_results.extend(
            results
        )

    if not all_results:
        print(
            "No experiments were run."
        )
        return

    output = (
        "homberger_family_benchmark.csv"
    )

    with open(
        output,
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
    print("=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)

    print(
        "Experiments saved:",
        len(all_results)
    )

    print(
        "Results:",
        output
    )


if __name__ == "__main__":
    main()
