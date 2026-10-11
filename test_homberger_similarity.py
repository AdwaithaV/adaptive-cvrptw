"""
test_homberger_similarity.py

Tests feature extraction, adaptive weights, and similarity
on a small subset of a Homberger CVRPTW instance.
"""

from pathlib import Path

from data.homberger_loader import load_homberger_instance
from graph.build_graph import build_graph
from algorithms.similarity import SimilarityCalculator
from algorithms.adaptive_weights import AdaptiveWeightCalculator


def main() -> None:
    dataset_path = Path(
        "data/datasets/homberger/C1_10_1.TXT"
    )

    instance = load_homberger_instance(dataset_path)

    customers = instance["customers"].head(100).copy()

    graph = build_graph(
        customers=customers,
        k=5,
        exclude_depot=True,
    )

    print("=" * 60)
    print("HOMBERGER SIMILARITY TEST")
    print("=" * 60)

    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    weight_calculator = AdaptiveWeightCalculator(graph)
    similarity_calculator = SimilarityCalculator(graph)

    first_node = next(iter(graph.nodes))
    first_edge = next(iter(graph.edges))

    alpha, beta = weight_calculator.adaptive_weights(
        first_node
    )

    adaptive_score = similarity_calculator.adaptive_similarity(
        first_edge[0],
        first_edge[1],
    )

    spatial_score = similarity_calculator.spatial_similarity(
        first_edge[0],
        first_edge[1],
    )

    temporal_score = similarity_calculator.temporal_similarity(
        first_edge[0],
        first_edge[1],
    )

    demand_score = similarity_calculator.demand_similarity(
        first_edge[0],
        first_edge[1],
    )

    print("\nFirst node:", first_node)
    print("Alpha:", alpha)
    print("Beta:", beta)

    print("\nFirst edge:", first_edge)
    print("Spatial similarity:", spatial_score)
    print("Temporal similarity:", temporal_score)
    print("Demand similarity:", demand_score)
    print("Adaptive similarity:", adaptive_score)

    assert graph.number_of_nodes() == 100
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= beta <= 1.0
    assert abs((alpha + beta) - 1.0) < 1e-9
    assert spatial_score >= 0.0
    assert temporal_score >= 0.0
    assert demand_score >= 0.0
    assert adaptive_score >= 0.0

    print("\nAll similarity checks passed.")


if __name__ == "__main__":
    main()

