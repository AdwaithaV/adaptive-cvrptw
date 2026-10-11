"""
test_homberger_adaptive_coarsening.py

Full 100 -> 50 adaptive coarsening experiment
on real Homberger CVRPTW data.
"""

from pathlib import Path

import networkx as nx

from algorithms.adaptive_coarsening import (
    AdaptiveGraphCoarsener,
)
from data.homberger_loader import (
    load_homberger_instance,
)
from graph.build_graph import (
    build_graph,
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
        .head(100)
        .copy()
    )

    graph = build_graph(
        customers=customers,
        k=5,
        exclude_depot=True,
    )

    print(
        "\n# HOMBERGER ADAPTIVE "
        "COARSENING TEST\n"
    )

    coarsener = (
        AdaptiveGraphCoarsener(
            graph=graph,
            vehicle_capacity=float(
                instance[
                    "vehicle_capacity"
                ]
            ),
            require_sequence_feasibility=True,
        )
    )

    coarsened_graph = (
        coarsener.run(
            target_ratio=0.50,
            verbose=True,
        )
    )

    validation = (
        coarsener.validate()
    )

    coarsener.summary()

    print()
    print("Graph statistics")
    print("-" * 60)

    print(
        "Final edges:",
        coarsened_graph
        .number_of_edges(),
    )

    print(
        "Final density:",
        nx.density(
            coarsened_graph
        ),
    )

    average_degree = (
        sum(
            dict(
                coarsened_graph.degree()
            ).values()
        )
        /
        coarsened_graph
        .number_of_nodes()
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
            coarsened_graph.nodes[
                node
            ]["members"]
        )
        for node in (
            coarsened_graph.nodes
        )
    ]

    print(
        "Largest meta-node:",
        max(cluster_sizes),
    )

    print(
        "Average meta-node size:",
        sum(cluster_sizes)
        / len(cluster_sizes),
    )

    assert (
        validation[
            "coarsened_nodes"
        ]
        == 50
    ), (
        "Adaptive coarsener "
        "did not reach 50 nodes."
    )

    assert validation[
        "connected"
    ]

    assert validation[
        "membership_preserved"
    ]

    assert not validation[
        "duplicate_members"
    ]

    member_information_preserved = all(
        (
            len(
                coarsened_graph.nodes[node].get(
                    "member_windows",
                    {}
                )
            )
            == len(
                coarsened_graph.nodes[node]["members"]
            )
        )
        or (
            len(
                coarsened_graph.nodes[node]["members"]
            ) == 1
            and "ready_time" in coarsened_graph.nodes[node]
            and "due_time" in coarsened_graph.nodes[node]
        )
        for node in coarsened_graph.nodes
    )

    assert member_information_preserved

    assert validation[
        "demand_preserved"
    ]

    assert validation[
        "capacity_valid"
    ]

    print()
    print(
        "All adaptive "
        "coarsening checks passed."
    )


if __name__ == "__main__":
    main()

