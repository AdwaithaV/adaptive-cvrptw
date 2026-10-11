"""
similarity.py

Similarity computation for adaptive graph coarsening.

Includes:
1. Spatial similarity
2. Temporal similarity
3. Time-window overlap
4. Demand similarity
5. Sequence compatibility
6. Adaptive combined similarity
"""

from __future__ import annotations

import math

from algorithms.adaptive_weights import AdaptiveWeightCalculator


class SimilarityCalculator:

    def __init__(self, graph):

        self.G = graph

        self.weight_calculator = (
            AdaptiveWeightCalculator(graph)
        )

    def spatial_distance(
        self,
        node1,
        node2,
    ):

        x1 = self.G.nodes[node1]["x"]
        y1 = self.G.nodes[node1]["y"]

        x2 = self.G.nodes[node2]["x"]
        y2 = self.G.nodes[node2]["y"]

        return math.sqrt(
            (x1 - x2) ** 2
            + (y1 - y2) ** 2
        )

    def spatial_similarity(
        self,
        node1,
        node2,
    ):

        distance = self.spatial_distance(
            node1,
            node2,
        )

        return 1.0 / (
            1.0 + distance
        )

    def midpoint(self, node):

        ready = float(
            self.G.nodes[node][
                "ready_time"
            ]
        )

        due = float(
            self.G.nodes[node][
                "due_time"
            ]
        )

        return (
            ready + due
        ) / 2.0

    def temporal_distance(
        self,
        node1,
        node2,
    ):

        return abs(
            self.midpoint(node1)
            - self.midpoint(node2)
        )

    def temporal_similarity(
        self,
        node1,
        node2,
    ):

        distance = (
            self.temporal_distance(
                node1,
                node2,
            )
        )

        return 1.0 / (
            1.0 + distance
        )

    def time_overlap(
        self,
        node1,
        node2,
    ):

        r1 = float(
            self.G.nodes[node1][
                "ready_time"
            ]
        )

        d1 = float(
            self.G.nodes[node1][
                "due_time"
            ]
        )

        r2 = float(
            self.G.nodes[node2][
                "ready_time"
            ]
        )

        d2 = float(
            self.G.nodes[node2][
                "due_time"
            ]
        )

        overlap = max(
            0.0,
            min(d1, d2)
            - max(r1, r2),
        )

        union = (
            max(d1, d2)
            - min(r1, r2)
        )

        if union <= 0:
            return 0.0

        return overlap / union

    def demand_similarity(
        self,
        node1,
        node2,
    ):

        d1 = float(
            self.G.nodes[node1][
                "demand"
            ]
        )

        d2 = float(
            self.G.nodes[node2][
                "demand"
            ]
        )

        return 1.0 / (
            1.0 + abs(d1 - d2)
        )

    def directional_sequence_slack(
        self,
        first_node,
        second_node,
    ):
        """
        Estimate remaining time-window slack if first_node
        is visited before second_node.

        Positive slack means the ordering is feasible.
        """

        first = self.G.nodes[
            first_node
        ]

        second = self.G.nodes[
            second_node
        ]

        travel_time = (
            self.spatial_distance(
                first_node,
                second_node,
            )
        )

        first_start = float(
            first["ready_time"]
        )

        first_finish = (
            first_start
            + float(
                first["service_time"]
            )
        )

        arrival_second = (
            first_finish
            + travel_time
        )

        second_start = max(
            arrival_second,
            float(
                second["ready_time"]
            ),
        )

        slack = (
            float(
                second["due_time"]
            )
            - second_start
        )

        return slack

    def sequence_compatibility(
        self,
        node1,
        node2,
    ):
        """
        Continuous sequence compatibility score.

        Tests both:
            node1 -> node2
            node2 -> node1

        Uses the better ordering.

        Score is in [0, 1].

        Larger positive scheduling slack gives a score
        closer to 1. Infeasible pairs receive 0.
        """

        forward_slack = (
            self.directional_sequence_slack(
                node1,
                node2,
            )
        )

        backward_slack = (
            self.directional_sequence_slack(
                node2,
                node1,
            )
        )

        best_slack = max(
            forward_slack,
            backward_slack,
        )

        if best_slack < 0:
            return 0.0

        first_width = max(
            0.0,
            float(
                self.G.nodes[node1][
                    "due_time"
                ]
            )
            - float(
                self.G.nodes[node1][
                    "ready_time"
                ]
            ),
        )

        second_width = max(
            0.0,
            float(
                self.G.nodes[node2][
                    "due_time"
                ]
            )
            - float(
                self.G.nodes[node2][
                    "ready_time"
                ]
            ),
        )

        scale = max(
            1.0,
            first_width,
            second_width,
        )

        normalized_slack = (
            best_slack / scale
        )

        return normalized_slack / (
            1.0 + normalized_slack
        )

    def adaptive_similarity(
        self,
        node1,
        node2,
        sequence_weight=0.0,
        demand_weight=0.05,
    ):
        """
        Adaptive similarity with optional sequence term.

        Spatial and temporal weights share the remaining
        weight after demand and sequence terms.
        """

        if sequence_weight < 0:
            raise ValueError(
                "sequence_weight must be non-negative."
            )

        if demand_weight < 0:
            raise ValueError(
                "demand_weight must be non-negative."
            )

        if (
            sequence_weight
            + demand_weight
            >= 1.0
        ):
            raise ValueError(
                "sequence_weight + demand_weight "
                "must be less than 1."
            )

        alpha, beta = (
            self.weight_calculator
            .edge_weights(
                node1,
                node2,
            )
        )

        remaining_weight = (
            1.0
            - sequence_weight
            - demand_weight
        )

        spatial_weight = (
            remaining_weight
            * alpha
        )

        temporal_weight = (
            remaining_weight
            * beta
        )

        spatial = (
            self.spatial_similarity(
                node1,
                node2,
            )
        )

        temporal = (
            0.5
            * self.temporal_similarity(
                node1,
                node2,
            )
            +
            0.5
            * self.time_overlap(
                node1,
                node2,
            )
        )

        demand = (
            self.demand_similarity(
                node1,
                node2,
            )
        )

        sequence = (
            self.sequence_compatibility(
                node1,
                node2,
            )
        )

        score = (
            spatial_weight
            * spatial

            + temporal_weight
            * temporal

            + demand_weight
            * demand

            + sequence_weight
            * sequence
        )

        return float(score)
