"""
test_homberger_fixed_500.py

Fixed-weight CVRPTW graph coarsening experiment.

500 real Homberger customers -> 250 meta-nodes.
"""

from pathlib import Path

import networkx as nx

from algorithms.fixed_coarsening import FixedGraphCoarsener
from data.homberger_loader import load_homberger_instance
from graph.build_graph import build_graph


def main():

    dataset_path = Path(
        "data/datasets/homberger/C1_10_1.TXT"
    )

    instance = load_homberger_instance(
        dataset_path
    )

    customers = (
        instance["customers"]
        .head(500)
        .copy()
    )

    print("=" * 70)
    print("HOMBERGER 500-CUSTOMER FIXED COARSENING")
    print("=" * 70)

    print("\nBuilding 500-customer graph...")

    graph = build_graph(
        customers=customers,
        k=5,
        exclude_depot=True,
    )

    print("Original nodes:", graph.number_of_nodes())
    print("Original edges:", graph.number_of_edges())
    print("Connected:", nx.is_connected(graph))
    print("Original density:", nx.density(graph))

    original_average_degree = (
        sum(dict(graph.degree()).values())
        / graph.number_of_nodes()
    )

    original_clustering = (
        nx.average_clustering(graph)
    )

    print(
        "Original average degree:",
        original_average_degree,
    )

    print(
        "Original clustering:",
        original_clustering,
    )

    print("\nStarting fixed coarsening...\n")

    coarsener = FixedGraphCoarsener(
        graph=graph,
        vehicle_capacity=float(
            instance["vehicle_capacity"]
        ),
        require_sequence_feasibility=True,
        alpha=0.5,
        beta=0.5,
        gamma=0.10,
    )

    coarsened_graph = coarsener.run(
        target_ratio=0.50,
        verbose=False,
    )

    validation = coarsener.validate()

    print()
    print("=" * 70)
    print("FIXED 500 -> 250 RESULTS")
    print("=" * 70)

    print(
        "Original nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Coarsened nodes:",
        coarsened_graph.number_of_nodes(),
    )

    print(
        "Successful contractions:",
        coarsener.total_successful_contractions,
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
        "Runtime seconds:",
        coarsener.runtime_seconds,
    )

    print(
        "Connected:",
        nx.is_connected(coarsened_graph),
    )

    print(
        "Membership preserved:",
        validation["membership_preserved"],
    )

    print(
        "Duplicate members:",
        validation["duplicate_members"],
    )

    print(
        "Demand preserved:",
        validation["demand_preserved"],
    )

    print(
        "Capacity valid:",
        validation["capacity_valid"],
    )

    print()
    print("GRAPH STATISTICS")
    print("-" * 70)

    final_edges = (
        coarsened_graph.number_of_edges()
    )

    final_density = nx.density(
        coarsened_graph
    )

    final_average_degree = (
        sum(
            dict(
                coarsened_graph.degree()
            ).values()
        )
        / coarsened_graph.number_of_nodes()
    )

    final_clustering = (
        nx.average_clustering(
            coarsened_graph
        )
    )

    cluster_sizes = [
        len(
            coarsened_graph.nodes[
                node
            ]["members"]
        )
        for node in coarsened_graph.nodes
    ]

    print(
        "Final edges:",
        final_edges,
    )

    print(
        "Final density:",
        final_density,
    )

    print(
        "Final average degree:",
        final_average_degree,
    )

    print(
        "Final clustering:",
        final_clustering,
    )

    print(
        "Largest meta-node:",
        max(cluster_sizes),
    )

    print(
        "Average meta-node size:",
        sum(cluster_sizes)
        / len(cluster_sizes),
    )

    print()
    print("PRESERVATION")
    print("-" * 70)

    print(
        "Degree change:",
        final_average_degree
        - original_average_degree,
    )

    print(
        "Clustering change:",
        final_clustering
        - original_clustering,
    )

    assert (
        coarsened_graph.number_of_nodes()
        == 250
    )

    assert validation["connected"]
    assert validation["membership_preserved"]
    assert not validation["duplicate_members"]
    assert validation["demand_preserved"]
    assert validation["capacity_valid"]

    print()
    print(
        "ALL 500-CUSTOMER FIXED "
        "COARSENING CHECKS PASSED."
    )


if __name__ == "__main__":
    main()
