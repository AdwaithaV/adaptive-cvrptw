"""
sweep_sequence_weight_500.py

Evaluate sequence-aware Adaptive v2 on 500 Homberger customers.
"""

from pathlib import Path

from algorithms.adaptive_coarsening import AdaptiveGraphCoarsener
from data.homberger_loader import load_homberger_instance
from graph.build_graph import build_graph
from metrics.cluster_quality import ClusterQualityEvaluator


def main():

    instance = load_homberger_instance(
        Path(
            "data/datasets/homberger/C1_10_1.TXT"
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
        instance["vehicle_capacity"]
    )

    weights = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ]

    print("=" * 100)
    print("SEQUENCE-WEIGHT SWEEP - HOMBERGER 500 -> 250")
    print("=" * 100)

    print(
        f"{'Weight':>8} "
        f"{'Runtime':>10} "
        f"{'Spatial':>12} "
        f"{'Temporal':>12} "
        f"{'SeqRate':>10} "
        f"{'Largest':>10}"
    )

    print("-" * 100)

    for weight in weights:

        coarsener = AdaptiveGraphCoarsener(
            graph=graph,
            vehicle_capacity=capacity,
            require_sequence_feasibility=True,
            sequence_weight=weight,
        )

        result_graph = coarsener.run(
            target_ratio=0.50,
            verbose=False,
        )

        quality = ClusterQualityEvaluator(
            result_graph,
            capacity,
        ).evaluate()

        print(
            f"{weight:8.2f} "
            f"{coarsener.runtime_seconds:10.3f} "
            f"{quality['mean_spatial_compactness']:12.3f} "
            f"{quality['mean_temporal_dispersion']:12.3f} "
            f"{quality['sequence_feasibility_rate']:10.3f} "
            f"{quality['largest_cluster']:10d}"
        )


if __name__ == "__main__":
    main()


