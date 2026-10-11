"""
compare.py

Comparison plots between
Fixed and Adaptive graph coarsening.
"""

import matplotlib.pyplot as plt
import pandas as pd


class ComparisonVisualizer:

    @staticmethod
    def plot_metrics(metrics_df):
        """
        Plot comparison metrics.

        metrics_df should contain:

        Metric
        Fixed
        Adaptive
        """

        metrics = metrics_df["Metric"]

        fixed = metrics_df["Fixed"]

        adaptive = metrics_df["Adaptive"]

        x = range(len(metrics))

        width = 0.35

        plt.figure(figsize=(12, 6))

        plt.bar(
            [i - width / 2 for i in x],
            fixed,
            width,
            label="Fixed"
        )

        plt.bar(
            [i + width / 2 for i in x],
            adaptive,
            width,
            label="Adaptive"
        )

        plt.xticks(x, metrics, rotation=30)

        plt.ylabel("Metric Value")

        plt.title("Fixed vs Adaptive Graph Coarsening")

        plt.legend()

        plt.tight_layout()

        plt.show()

    @staticmethod
    def similarity_histogram(graph):
        """
        Histogram of edge similarities.
        """

        similarities = [
            graph[u][v].get("similarity", 0)
            for u, v in graph.edges()
        ]

        plt.figure(figsize=(8, 5))

        plt.hist(
            similarities,
            bins=20
        )

        plt.xlabel("Similarity")

        plt.ylabel("Frequency")

        plt.title("Distribution of Edge Similarity")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def meta_node_sizes(history):
        """
        Plot distribution of meta-node sizes.

        history:
            contraction history
        """

        sizes = [len(group) for group in history]

        plt.figure(figsize=(8, 5))

        plt.hist(
            sizes,
            bins=15
        )

        plt.xlabel("Meta-node Size")

        plt.ylabel("Frequency")

        plt.title("Meta-node Size Distribution")

        plt.tight_layout()

        plt.show()