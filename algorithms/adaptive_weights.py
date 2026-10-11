"""
adaptive_weights.py

Cached adaptive weight computation for CVRPTW graph coarsening.

Graph statistics are computed once per calculator instance,
then reused for all node and edge weight calculations.
"""

from __future__ import annotations

import networkx as nx

from graph.graph_utils import GraphUtils


class AdaptiveWeightCalculator:

    def __init__(self, graph: nx.Graph):

        self.G = graph
        self.utils = GraphUtils(graph)

        self._feature_cache = {}
        self._weight_cache = {}

        self._statistics = (
            self._compute_graph_statistics()
        )

    def normalize(
        self,
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        if maximum - minimum == 0:
            return 0.0

        return (
            value - minimum
        ) / (
            maximum - minimum
        )

    def feature_vector(self, node):
        """
        Return a cached feature vector.
        """

        if node not in self._feature_cache:
            self._feature_cache[node] = (
                self.utils.feature_vector(node)
            )

        return self._feature_cache[node]

    def _compute_graph_statistics(self):
        """
        Compute graph-wide min/max statistics once.
        """

        density = []
        degree = []
        clustering = []

        spacing = []
        width = []
        timediff = []

        for node in self.G.nodes:

            features = (
                self.feature_vector(node)
            )

            density.append(
                features["density"]
            )

            degree.append(
                features["degree"]
            )

            clustering.append(
                features["clustering"]
            )

            spacing.append(
                features["avg_distance"]
            )

            width.append(
                features["window_width"]
            )

            timediff.append(
                features[
                    "avg_time_difference"
                ]
            )

        return {
            "density": (
                min(density),
                max(density),
            ),

            "degree": (
                min(degree),
                max(degree),
            ),

            "clustering": (
                min(clustering),
                max(clustering),
            ),

            "spacing": (
                min(spacing),
                max(spacing),
            ),

            "width": (
                min(width),
                max(width),
            ),

            "timediff": (
                min(timediff),
                max(timediff),
            ),
        }

    def compute_alpha(self, node):

        stats = self._statistics

        features = (
            self.feature_vector(node)
        )

        density = self.normalize(
            features["density"],
            *stats["density"],
        )

        degree = self.normalize(
            features["degree"],
            *stats["degree"],
        )

        clustering = self.normalize(
            features["clustering"],
            *stats["clustering"],
        )

        spacing = self.normalize(
            features["avg_distance"],
            *stats["spacing"],
        )

        alpha = (
            0.35 * density
            + 0.30 * degree
            + 0.20 * clustering
            + 0.15 * (1.0 - spacing)
        )

        return float(alpha)

    def compute_beta(self, node):

        stats = self._statistics

        features = (
            self.feature_vector(node)
        )

        width = self.normalize(
            features["window_width"],
            *stats["width"],
        )

        timediff = self.normalize(
            features[
                "avg_time_difference"
            ],
            *stats["timediff"],
        )

        beta = (
            0.60 * (1.0 - width)
            + 0.40 * (
                1.0 - timediff
            )
        )

        return float(beta)

    def adaptive_weights(self, node):
        """
        Return cached normalized alpha and beta.
        """

        if node in self._weight_cache:
            return self._weight_cache[node]

        alpha = self.compute_alpha(node)
        beta = self.compute_beta(node)

        total = alpha + beta

        if total == 0:
            result = (0.5, 0.5)

        else:
            result = (
                alpha / total,
                beta / total,
            )

        self._weight_cache[node] = result

        return result

    def edge_weights(
        self,
        first_node,
        second_node,
    ):

        first_alpha, first_beta = (
            self.adaptive_weights(
                first_node
            )
        )

        second_alpha, second_beta = (
            self.adaptive_weights(
                second_node
            )
        )

        alpha = (
            first_alpha
            + second_alpha
        ) / 2.0

        beta = (
            first_beta
            + second_beta
        ) / 2.0

        return (
            float(alpha),
            float(beta),
        )
