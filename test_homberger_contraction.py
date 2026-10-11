"""
test_homberger_contraction.py

Tests one feasible CVRPTW graph contraction using real
Homberger benchmark data.
"""

from pathlib import Path

import networkx as nx

from algorithms.contraction import (
    contract_customer_nodes,
    contraction_feasible,
)
from algorithms.similarity import SimilarityCalculator
from data.homberger_loader import load_homberger_instance
from graph.build_graph import build_graph


def main() -> None:
    dataset_path = Path(
        "data/datasets/homberger/C1_10_1.TXT"
    )

    instance = load_homberger_instance(
        dataset_path
    )

    customers = instance["customers"].head(100).copy()

    graph = build_graph(
        customers=customers,
        k=5,
        exclude_depot=True,
    )

    vehicle_capacity = float(
        instance["vehicle_capacity"]
    )

    similarity_calculator = SimilarityCalculator(
        graph
    )

    feasible_edges = []

    for first_node, second_node in graph.edges:
        feasible, _ = contraction_feasible(
            graph=graph,
            first_node=first_node,
            second_node=second_node,
            vehicle_capacity=vehicle_capacity,
            require_time_overlap=True,
        )

        if not feasible:
            continue

        score = (
            similarity_calculator.adaptive_similarity(
                first_node,
                second_node,
            )
        )

        feasible_edges.append(
            (
                score,
                first_node,
                second_node,
            )
        )

    if not feasible_edges:
        raise RuntimeError(
            "No feasible contraction edge was found."
        )

    feasible_edges.sort(reverse=True)

    best_score, first_node, second_node = (
        feasible_edges[0]
    )

    original_node_count = graph.number_of_nodes()
    original_total_demand = sum(
        float(graph.nodes[node]["demand"])
        for node in graph.nodes
    )

    original_members = {
        member
        for node in graph.nodes
        for member in graph.nodes[node]["members"]
    }

    first_demand = float(
        graph.nodes[first_node]["demand"]
    )

    second_demand = float(
        graph.nodes[second_node]["demand"]
    )

    first_members = list(
        graph.nodes[first_node]["members"]
    )

    second_members = list(
        graph.nodes[second_node]["members"]
    )

    contracted_graph = contract_customer_nodes(
        graph=graph,
        first_node=first_node,
        second_node=second_node,
        vehicle_capacity=vehicle_capacity,
        require_time_overlap=True,
    )

    contracted_total_demand = sum(
        float(contracted_graph.nodes[node]["demand"])
        for node in contracted_graph.nodes
    )

    contracted_members = {
        member
        for node in contracted_graph.nodes
        for member in contracted_graph.nodes[node]["members"]
    }

    merged_data = contracted_graph.nodes[
        first_node
    ]

    print("=" * 60)
    print("HOMBERGER CONTRACTION TEST")
    print("=" * 60)

    print("Selected edge:", (first_node, second_node))
    print("Adaptive similarity:", best_score)

    print("\nOriginal nodes:", original_node_count)
    print(
        "Contracted nodes:",
        contracted_graph.number_of_nodes(),
    )

    print("\nFirst demand:", first_demand)
    print("Second demand:", second_demand)
    print("Merged demand:", merged_data["demand"])

    print("\nFirst members:", first_members)
    print("Second members:", second_members)
    print("Merged members:", merged_data["members"])

    print(
        "\nMerged time window:",
        (
            merged_data["ready_time"],
            merged_data["due_time"],
        ),
    )

    print(
        "Merged position:",
        merged_data["position"],
    )

    print(
        "Graph connected:",
        nx.is_connected(contracted_graph),
    )

    assert contracted_graph.number_of_nodes() == (
        original_node_count - 1
    )

    assert second_node not in contracted_graph

    assert merged_data["demand"] == (
        first_demand + second_demand
    )

    assert set(merged_data["members"]) == (
        set(first_members)
        | set(second_members)
    )

    assert abs(
        original_total_demand
        - contracted_total_demand
    ) < 1e-9

    assert original_members == contracted_members

    assert merged_data["demand"] <= vehicle_capacity

    assert (
        merged_data["ready_time"]
        <= merged_data["due_time"]
    )

    assert nx.is_connected(contracted_graph)

    assert all(
        "distance" in contracted_graph.edges[edge]
        for edge in contracted_graph.edges
    )

    print("\nAll contraction checks passed.")


if __name__ == "__main__":
    main()
