"""
test_homberger_fixed_coarsening.py

100 -> 50 fixed-weight baseline experiment
using real Homberger CVRPTW data.
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
        .head(100)
        .copy()
    )

    graph = build_graph(
        customers=customers,
        k=5,
        exclude_depot=True,
    )

    print("\n# HOMBERGER FIXED COARSENING TEST\n")

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
        verbose=True,
    )

    validation = coarsener.validate()

    coarsener.summary()

    print()
    print("Graph statistics")
    print("-" * 60)

    print(
        "Final edges:",
        coarsened_graph.number_of_edges(),
    )

    print(
        "Final density:",
        nx.density(coarsened_graph),
    )

    average_degree = (
        sum(
            dict(
                coarsened_graph.degree()
            ).values()
        )
        / coarsened_graph.number_of_nodes()
    )

    print(
        "Final average degree:",
        average_degree,
    )

    print(
        "Final clustering:",
        nx.average_clustering(
            coarsened_graph
        ),
    )

    cluster_sizes = [
        len(
            coarsened_graph.nodes[node]["members"]
        )
        for node in coarsened_graph.nodes
    ]

    print(
        "Largest meta-node:",
        max(cluster_sizes),
    )

    print(
        "Average meta-node size:",
        sum(cluster_sizes) / len(cluster_sizes),
    )

    assert validation["coarsened_nodes"] == 50
    assert validation["connected"]
    assert validation["membership_preserved"]
    assert not validation["duplicate_members"]
    assert validation["demand_preserved"]
    assert validation["capacity_valid"]

    print()
    print(
        "All fixed coarsening checks passed."
    )


if __name__ == "__main__":
    main()
