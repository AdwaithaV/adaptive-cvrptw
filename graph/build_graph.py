"""Graph construction utilities for CVRPTW datasets."""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "x",
    "y",
    "demand",
    "ready_time",
    "due_time",
    "service_time",
}


def validate_customer_data(customers: pd.DataFrame) -> None:
    """
    Validate that a customer DataFrame contains all required CVRPTW columns.

    Parameters
    ----------
    customers:
        DataFrame containing customer information.

    Raises
    ------
    TypeError
        If customers is not a pandas DataFrame.
    ValueError
        If required columns are missing or the DataFrame is empty.
    """

    if not isinstance(customers, pd.DataFrame):
        raise TypeError(
            "customers must be a pandas DataFrame."
        )

    if customers.empty:
        raise ValueError("Customer DataFrame is empty.")

    missing_columns = REQUIRED_COLUMNS.difference(customers.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Customer DataFrame is missing columns: {missing_text}"
        )

    if customers["customer_id"].duplicated().any():
        duplicated_ids = customers.loc[
            customers["customer_id"].duplicated(),
            "customer_id",
        ].tolist()

        raise ValueError(
            f"Duplicate customer IDs found: {duplicated_ids[:10]}"
        )

    numeric_columns = [
        "customer_id",
        "x",
        "y",
        "demand",
        "ready_time",
        "due_time",
        "service_time",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(customers[column]):
            raise ValueError(
                f"Column '{column}' must contain numeric values."
            )

    if customers[numeric_columns].isnull().any().any():
        raise ValueError("Customer data contains missing numeric values.")

    invalid_time_windows = (
        customers["ready_time"] > customers["due_time"]
    )

    if invalid_time_windows.any():
        invalid_ids = customers.loc[
            invalid_time_windows,
            "customer_id",
        ].tolist()

        raise ValueError(
            "ready_time is greater than due_time for customers: "
            f"{invalid_ids[:10]}"
        )

    if (customers["demand"] < 0).any():
        raise ValueError("Customer demand cannot be negative.")


def euclidean_distance(
    first_position: Iterable[float],
    second_position: Iterable[float],
) -> float:
    """Return Euclidean distance between two 2D positions."""

    first = np.asarray(first_position, dtype=float)
    second = np.asarray(second_position, dtype=float)

    return float(np.linalg.norm(first - second))


def build_knn_graph(
    customers: pd.DataFrame,
    k: int = 5,
    exclude_depot: bool = True,
) -> nx.Graph:
    """
    Build an undirected k-nearest-neighbour graph from CVRPTW customers.

    Each graph node stores:

    - x
    - y
    - position
    - demand
    - ready_time
    - due_time
    - service_time
    - time_window_width
    - is_depot
    - members

    Each graph edge initially stores:

    - distance

    Similarity values can later be added by the similarity module.

    Parameters
    ----------
    customers:
        CVRPTW customer DataFrame.

    k:
        Number of nearest neighbours considered for every node.

    exclude_depot:
        If True, customer 0 is removed before constructing the graph.
        This is recommended for graph coarsening.

    Returns
    -------
    networkx.Graph
        Undirected customer graph.
    """

    validate_customer_data(customers)

    graph_data = customers.copy()

    if exclude_depot:
        graph_data = graph_data[
            graph_data["customer_id"] != 0
        ].copy()

    graph_data = graph_data.reset_index(drop=True)

    number_of_nodes = len(graph_data)

    if number_of_nodes == 0:
        raise ValueError(
            "No customer nodes remain after depot removal."
        )

    if k < 1:
        raise ValueError("k must be at least 1.")

    if number_of_nodes == 1:
        effective_k = 0
    else:
        effective_k = min(k, number_of_nodes - 1)

    graph = nx.Graph()

    # Add customer nodes.
    for row in graph_data.itertuples(index=False):
        customer_id = int(row.customer_id)
        x = float(row.x)
        y = float(row.y)
        ready_time = float(row.ready_time)
        due_time = float(row.due_time)

        graph.add_node(
            customer_id,
            customer_id=customer_id,
            x=x,
            y=y,
            position=(x, y),
            demand=float(row.demand),
            ready_time=ready_time,
            due_time=due_time,
            service_time=float(row.service_time),
            time_window_width=due_time - ready_time,
            is_depot=customer_id == 0,
            members=[customer_id],
        )

    if effective_k == 0:
        return graph

    coordinates = graph_data[["x", "y"]].to_numpy(
        dtype=float
    )

    customer_ids = graph_data["customer_id"].astype(int).to_numpy()

    # Pairwise Euclidean distances.
    coordinate_differences = (
        coordinates[:, np.newaxis, :]
        - coordinates[np.newaxis, :, :]
    )

    distance_matrix = np.linalg.norm(
        coordinate_differences,
        axis=2,
    )

    # Add each customer's k closest neighbours.
    for row_index, customer_id in enumerate(customer_ids):
        sorted_indices = np.argsort(distance_matrix[row_index])

        neighbour_indices = [
            index
            for index in sorted_indices
            if index != row_index
        ][:effective_k]

        for neighbour_index in neighbour_indices:
            neighbour_id = int(customer_ids[neighbour_index])
            distance = float(
                distance_matrix[row_index, neighbour_index]
            )

            if graph.has_edge(customer_id, neighbour_id):
                continue

            graph.add_edge(
                customer_id,
                neighbour_id,
                distance=distance,
            )

    return graph
def connect_graph_components(graph: nx.Graph) -> nx.Graph:
    """
    Connect disconnected components using the nearest pair of nodes.

    The graph is modified in place and always returned.
    """

    if graph.number_of_nodes() <= 1:
        return graph

    while not nx.is_connected(graph):
        components = [
            list(component)
            for component in nx.connected_components(graph)
        ]

        best_pair = None
        best_distance = float("inf")

        for first_index in range(len(components)):
            for second_index in range(
                first_index + 1,
                len(components),
            ):
                first_component = components[first_index]
                second_component = components[second_index]

                for first_node in first_component:
                    first_position = graph.nodes[
                        first_node
                    ]["position"]

                    for second_node in second_component:
                        second_position = graph.nodes[
                            second_node
                        ]["position"]

                        distance = euclidean_distance(
                            first_position,
                            second_position,
                        )

                        if distance < best_distance:
                            best_distance = distance
                            best_pair = (
                                first_node,
                                second_node,
                            )

        if best_pair is None:
            raise RuntimeError(
                "Unable to connect graph components."
            )

        graph.add_edge(
            best_pair[0],
            best_pair[1],
            distance=best_distance,
            component_bridge=True,
        )

    return graph

def build_graph(
    customers: pd.DataFrame,
    k: int = 5,
    exclude_depot: bool = True,
) -> nx.Graph:
    """
    Build a connected k-nearest-neighbour graph.

    First, construct the customer k-NN graph. Then connect any
    disconnected components using the closest pair of nodes
    between components.
    """

    graph = build_knn_graph(
        customers=customers,
        k=k,
        exclude_depot=exclude_depot,
    )

    graph = connect_graph_components(graph)

    return graph

