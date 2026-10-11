"""
test_homberger_graph.py

Verification script for loading a Homberger CVRPTW instance
and constructing a connected customer graph.
"""

from pathlib import Path

import networkx as nx

from data.homberger_loader import load_homberger_instance
from graph.build_graph import build_graph


def main() -> None:
    """
    Load one Homberger instance, build its customer graph,
    print graph statistics, and validate basic properties.
    """

    dataset_path = Path(
        "data/datasets/homberger/C1_10_1.TXT"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path.resolve()}"
        )

    instance = load_homberger_instance(dataset_path)

    graph = build_graph(
        customers=instance["customers"],
        k=5,
        exclude_depot=True,
    )

    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()

    if number_of_nodes == 0:
        raise RuntimeError(
            "The graph contains no customer nodes."
        )

    connected = nx.is_connected(graph)
    component_count = nx.number_connected_components(graph)
    average_degree = (
        sum(dict(graph.degree()).values())
        / number_of_nodes
    )

    print("=" * 60)
    print("HOMBERGER GRAPH VERIFICATION")
    print("=" * 60)

    print("Instance:", instance["instance_name"])
    print("Maximum vehicles:", instance["vehicle_count"])
    print("Vehicle capacity:", instance["vehicle_capacity"])
    print("Customers loaded:", len(instance["customers"]))
    print("Graph nodes:", number_of_nodes)
    print("Graph edges:", number_of_edges)
    print("Connected:", connected)
    print("Connected components:", component_count)
    print("Average degree:", average_degree)
    print("Density:", nx.density(graph))

    first_node = next(iter(graph.nodes))

    print("\nFirst node ID:", first_node)
    print("First node attributes:")
    print(graph.nodes[first_node])

    if number_of_edges > 0:
        first_edge = next(iter(graph.edges))

        print("\nFirst edge:", first_edge)
        print("First edge attributes:")
        print(graph.edges[first_edge])

    assert number_of_nodes == 1000, (
        f"Expected 1000 customer nodes, found {number_of_nodes}."
    )

    assert 0 not in graph.nodes, (
        "Depot node 0 should not be present."
    )

    assert connected, (
        f"Graph is disconnected and has "
        f"{component_count} connected components."
    )

    assert all(
        "distance" in graph.edges[edge]
        for edge in graph.edges
    ), "One or more edges do not contain distance data."

    assert all(
        "members" in graph.nodes[node]
        for node in graph.nodes
    ), "One or more nodes do not contain member information."

    required_node_attributes = {
        "customer_id",
        "x",
        "y",
        "position",
        "demand",
        "ready_time",
        "due_time",
        "service_time",
        "time_window_width",
        "is_depot",
        "members",
    }

    for node, attributes in graph.nodes(data=True):
        missing_attributes = (
            required_node_attributes
            - set(attributes.keys())
        )

        assert not missing_attributes, (
            f"Node {node} is missing attributes: "
            f"{sorted(missing_attributes)}"
        )

    print("\nAll Homberger graph checks passed.")


if __name__ == "__main__":
    main()
