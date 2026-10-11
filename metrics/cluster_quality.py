"""
cluster_quality.py

CVRPTW-specific quality metrics for coarsened meta-nodes.

Metrics
-------
1. Spatial compactness
2. Temporal dispersion
3. Capacity utilization
4. Internal greedy sequence feasibility
5. Cluster-size statistics
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import networkx as nx


def euclidean_distance(first, second) -> float:
    return math.sqrt(
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
    )


class ClusterQualityEvaluator:

    def __init__(
        self,
        graph: nx.Graph,
        vehicle_capacity: float,
    ) -> None:

        self.graph = graph
        self.vehicle_capacity = float(
            vehicle_capacity
        )

    def _members(self, node):
        return list(
            self.graph.nodes[node]["members"]
        )

    def _member_positions(self, node):
        data = self.graph.nodes[node]

        stored = data.get(
            "member_positions",
            {}
        )

        if stored:
            return stored

        return {
            member: tuple(
                data["position"]
            )
            for member in self._members(node)
        }

    def _member_windows(self, node):
        data = self.graph.nodes[node]

        stored = data.get(
            "member_windows",
            {}
        )

        if stored:
            return stored

        return {
            member: (
                float(data["ready_time"]),
                float(data["due_time"]),
            )
            for member in self._members(node)
        }

    def _member_service_times(self, node):
        data = self.graph.nodes[node]

        stored = data.get(
            "member_service_times",
            {}
        )

        if stored:
            return stored

        return {
            member: float(
                data["service_time"]
            )
            for member in self._members(node)
        }

    def spatial_compactness(self, node) -> float:
        """
        Average pairwise Euclidean distance among original
        customers inside a meta-node.

        Lower is better.
        """

        positions = (
            self._member_positions(node)
        )

        members = list(
            positions.keys()
        )

        if len(members) <= 1:
            return 0.0

        distances = []

        for first, second in combinations(
            members,
            2,
        ):
            distances.append(
                euclidean_distance(
                    positions[first],
                    positions[second],
                )
            )

        return float(
            np.mean(distances)
        )

    def temporal_dispersion(self, node) -> float:
        """
        Average pairwise difference between time-window
        midpoints of original members.

        Lower is better.
        """

        windows = (
            self._member_windows(node)
        )

        members = list(
            windows.keys()
        )

        if len(members) <= 1:
            return 0.0

        midpoints = {
            member: (
                windows[member][0]
                + windows[member][1]
            ) / 2.0
            for member in members
        }

        differences = []

        for first, second in combinations(
            members,
            2,
        ):
            differences.append(
                abs(
                    midpoints[first]
                    - midpoints[second]
                )
            )

        return float(
            np.mean(differences)
        )

    def capacity_utilization(self, node) -> float:
        """
        Fraction of vehicle capacity represented by cluster.
        """

        demand = float(
            self.graph.nodes[node]["demand"]
        )

        if self.vehicle_capacity <= 0:
            return 0.0

        return demand / self.vehicle_capacity

    def greedy_sequence_feasible(self, node) -> bool:
        """
        Approximate whether all original members of a cluster
        can be visited sequentially without violating their
        time windows.

        A nearest-feasible greedy scheduler is used.

        This is a heuristic feasibility indicator, not an
        exact CVRPTW route solver.
        """

        members = self._members(node)

        if len(members) <= 1:
            return True

        positions = (
            self._member_positions(node)
        )

        windows = (
            self._member_windows(node)
        )

        service_times = (
            self._member_service_times(node)
        )

        #
        # Try every member as a possible starting customer.
        #

        for start in members:

            unvisited = set(members)
            unvisited.remove(start)

            ready, due = windows[start]

            current_time = float(ready)

            if current_time > float(due):
                continue

            current_time += float(
                service_times[start]
            )

            current = start

            feasible = True

            while unvisited:

                candidates = []

                for next_member in unvisited:

                    travel = (
                        euclidean_distance(
                            positions[current],
                            positions[next_member],
                        )
                    )

                    arrival = (
                        current_time
                        + travel
                    )

                    next_ready, next_due = (
                        windows[next_member]
                    )

                    start_time = max(
                        arrival,
                        float(next_ready),
                    )

                    if start_time <= float(
                        next_due
                    ):
                        candidates.append(
                            (
                                start_time,
                                travel,
                                next_member,
                            )
                        )

                if not candidates:
                    feasible = False
                    break

                candidates.sort()

                (
                    service_start,
                    _,
                    selected,
                ) = candidates[0]

                current_time = (
                    service_start
                    + float(
                        service_times[
                            selected
                        ]
                    )
                )

                current = selected
                unvisited.remove(
                    selected
                )

            if feasible:
                return True

        return False

    def evaluate_node(self, node) -> dict:

        members = self._members(node)

        return {
            "node": node,
            "cluster_size":
                len(members),

            "spatial_compactness":
                self.spatial_compactness(
                    node
                ),

            "temporal_dispersion":
                self.temporal_dispersion(
                    node
                ),

            "capacity_utilization":
                self.capacity_utilization(
                    node
                ),

            "sequence_feasible":
                self.greedy_sequence_feasible(
                    node
                ),
        }

    def evaluate(self) -> dict:

        node_results = [
            self.evaluate_node(node)
            for node in self.graph.nodes
        ]

        non_singletons = [
            result
            for result in node_results
            if result["cluster_size"] > 1
        ]

        evaluation_set = (
            non_singletons
            if non_singletons
            else node_results
        )

        spatial_values = [
            result[
                "spatial_compactness"
            ]
            for result in evaluation_set
        ]

        temporal_values = [
            result[
                "temporal_dispersion"
            ]
            for result in evaluation_set
        ]

        utilization_values = [
            result[
                "capacity_utilization"
            ]
            for result in node_results
        ]

        feasible_clusters = sum(
            1
            for result in node_results
            if result[
                "sequence_feasible"
            ]
        )

        sizes = [
            result["cluster_size"]
            for result in node_results
        ]

        return {
            "num_clusters":
                len(node_results),

            "num_non_singleton_clusters":
                len(non_singletons),

            "mean_spatial_compactness":
                float(
                    np.mean(
                        spatial_values
                    )
                ),

            "median_spatial_compactness":
                float(
                    np.median(
                        spatial_values
                    )
                ),

            "mean_temporal_dispersion":
                float(
                    np.mean(
                        temporal_values
                    )
                ),

            "median_temporal_dispersion":
                float(
                    np.median(
                        temporal_values
                    )
                ),

            "mean_capacity_utilization":
                float(
                    np.mean(
                        utilization_values
                    )
                ),

            "max_capacity_utilization":
                float(
                    np.max(
                        utilization_values
                    )
                ),

            "sequence_feasible_clusters":
                feasible_clusters,

            "sequence_feasibility_rate":
                feasible_clusters
                / len(node_results),

            "largest_cluster":
                max(sizes),

            "average_cluster_size":
                float(
                    np.mean(sizes)
                ),

            "node_results":
                node_results,
        }
