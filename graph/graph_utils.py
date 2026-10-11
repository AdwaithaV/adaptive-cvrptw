"""
graph_utils.py

Graph-feature extraction utilities for adaptive graph coarsening.

Computes structural, spatial, temporal, and demand-related
features for every node in a CVRPTW graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


class GraphFeatureExtractor:
    """
    Extract graph and CVRPTW features for individual nodes.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.G = graph

    # ---------------------------------------------------------
    # Structural features
    # ---------------------------------------------------------

    def node_degree(self, node: int) -> int:
        return int(self.G.degree(node))

    def local_density(self, node: int) -> float:
        """
        Density among the neighbours of a node.
        """

        neighbours = list(self.G.neighbors(node))

        if len(neighbours) < 2:
            return 0.0

        neighbour_subgraph = self.G.subgraph(neighbours)

        return float(nx.density(neighbour_subgraph))

    def clustering_coefficient(self, node: int) -> float:
        return float(nx.clustering(self.G, node))

    # ---------------------------------------------------------
    # Spatial features
    # ---------------------------------------------------------

    def average_neighbor_distance(self, node: int) -> float:
        distances = []

        for neighbour in self.G.neighbors(node):
            edge_data = self.G.get_edge_data(
                node,
                neighbour,
                default={},
            )

            distance = edge_data.get("distance")

            if distance is None:
                first_position = self.G.nodes[node]["position"]
                second_position = self.G.nodes[neighbour]["position"]

                distance = float(
                    np.linalg.norm(
                        np.asarray(first_position, dtype=float)
                        - np.asarray(second_position, dtype=float)
                    )
                )

            distances.append(float(distance))

        return float(np.mean(distances)) if distances else 0.0

    # ---------------------------------------------------------
    # Temporal features
    # ---------------------------------------------------------

    def time_window_width(self, node: int) -> float:
        node_data = self.G.nodes[node]

        ready_time = float(node_data["ready_time"])
        due_time = float(node_data["due_time"])

        return max(0.0, due_time - ready_time)

    def time_window_midpoint(self, node: int) -> float:
        node_data = self.G.nodes[node]

        ready_time = float(node_data["ready_time"])
        due_time = float(node_data["due_time"])

        return (ready_time + due_time) / 2.0

    def average_neighbor_time_difference(
        self,
        node: int,
    ) -> float:
        node_midpoint = self.time_window_midpoint(node)

        differences = [
            abs(
                node_midpoint
                - self.time_window_midpoint(neighbour)
            )
            for neighbour in self.G.neighbors(node)
        ]

        return float(np.mean(differences)) if differences else 0.0

    # ---------------------------------------------------------
    # Demand features
    # ---------------------------------------------------------

    def demand(self, node: int) -> float:
        return float(self.G.nodes[node]["demand"])

    def average_neighbor_demand(self, node: int) -> float:
        values = [
            self.demand(neighbour)
            for neighbour in self.G.neighbors(node)
        ]

        return float(np.mean(values)) if values else 0.0

    # ---------------------------------------------------------
    # Complete feature vector
    # ---------------------------------------------------------

    def feature_vector(self, node: int) -> dict[str, float]:
        return {
            "degree": float(self.node_degree(node)),
            "density": self.local_density(node),
            "clustering": self.clustering_coefficient(node),
            "avg_distance": self.average_neighbor_distance(node),
            "window_width": self.time_window_width(node),
            "window_midpoint": self.time_window_midpoint(node),
            "avg_time_difference":
                self.average_neighbor_time_difference(node),
            "demand": self.demand(node),
            "neighbor_demand":
                self.average_neighbor_demand(node),
        }

    def compute_all_features(self) -> None:
        """
        Store computed features as node attributes.
        """

        for node in list(self.G.nodes):
            features = self.feature_vector(node)
            self.G.nodes[node].update(features)

    def print_features(self) -> None:
        print("\n========== GRAPH FEATURES ==========\n")

        for node in self.G.nodes:
            print(f"Node {node}")

            features = self.feature_vector(node)

            for key, value in features.items():
                print(f"{key:24s}: {value:.3f}")

            print()


# Backward-compatible name used by adaptive_weights.py.
GraphUtils = GraphFeatureExtractor
