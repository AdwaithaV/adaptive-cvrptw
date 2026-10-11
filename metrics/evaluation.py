"""
evaluation.py

Evaluation metrics for graph coarsening.

Compares

1. Fixed Graph Coarsening
2. Adaptive Graph Coarsening

Author: Adwaitha
"""

import numpy as np
import pandas as pd

from algorithms.similarity import (
    spatial_distance,
    temporal_distance
)


class GraphEvaluator:

    def __init__(self, dataframe):

        self.df = dataframe

    ########################################################
    # Reduction Ratio
    ########################################################

    def reduction_ratio(self, clusters):

        original = len(self.df) - 1
        reduced = len(clusters)

        return 100 * (1 - reduced / original)

    ########################################################
    # Average Cluster Size
    ########################################################

    def average_cluster_size(self, clusters):

        sizes = [len(c) for c in clusters]

        return np.mean(sizes)

    ########################################################
    # Cluster Diameter
    ########################################################

    def average_cluster_diameter(self, clusters):

        diameters = []

        for cluster in clusters:

            if len(cluster) == 1:

                diameters.append(0)

                continue

            distances = []

            for i in range(len(cluster)):

                for j in range(i + 1, len(cluster)):

                    c1 = self.df[
                        self.df["id"] == cluster[i]
                    ].iloc[0]

                    c2 = self.df[
                        self.df["id"] == cluster[j]
                    ].iloc[0]

                    distances.append(
                        spatial_distance(c1, c2)
                    )

            diameters.append(max(distances))

        return np.mean(diameters)

    ########################################################
    # Time Window Variance
    ########################################################

    def average_time_variance(self, clusters):

        variances = []

        for cluster in clusters:

            mids = []

            for cid in cluster:

                customer = self.df[
                    self.df["id"] == cid
                ].iloc[0]

                midpoint = (
                    customer["ready_time"]
                    +
                    customer["due_time"]
                ) / 2

                mids.append(midpoint)

            variances.append(np.var(mids))

        return np.mean(variances)

    ########################################################
    # Total Similarity
    ########################################################

    def average_similarity(self, clusters):

        scores = []

        for cluster in clusters:

            if len(cluster) == 1:

                continue

            c1 = self.df[
                self.df["id"] == cluster[0]
            ].iloc[0]

            c2 = self.df[
                self.df["id"] == cluster[1]
            ].iloc[0]

            spatial = spatial_distance(c1, c2)

            temporal = temporal_distance(c1, c2)

            scores.append(
                spatial + temporal
            )

        return np.mean(scores)

    ########################################################
    # Summary
    ########################################################

    def evaluate(self, clusters):

        results = {

            "Reduction Ratio (%)":
                self.reduction_ratio(clusters),

            "Average Cluster Size":
                self.average_cluster_size(clusters),

            "Average Diameter":
                self.average_cluster_diameter(clusters),

            "Average Time Variance":
                self.average_time_variance(clusters),

            "Average Similarity":
                self.average_similarity(clusters)

        }

        return results

    ########################################################
    # Pretty Print
    ########################################################

    def print_results(self, title, results):

        print()

        print("=" * 50)

        print(title)

        print("=" * 50)

        for key, value in results.items():

            print(f"{key:30s}: {value:.3f}")

        print()


###############################################################
# Standalone Testing
###############################################################

if __name__ == "__main__":

    from algorithms.fixed_coarsening import (
        FixedGraphCoarsener
    )

    from algorithms.adaptive_coarsening import (
        AdaptiveGraphCoarsener
    )

    df = pd.read_csv("synthetic_cvrptw.csv")

    evaluator = GraphEvaluator(df)

    ####################################################

    fixed = FixedGraphCoarsener(df)

    fixed_clusters = fixed.merge()

    fixed_results = evaluator.evaluate(
        fixed_clusters
    )

    evaluator.print_results(
        "Fixed Coarsening",
        fixed_results
    )

    ####################################################

    adaptive = AdaptiveGraphCoarsener(df)

    adaptive_clusters = adaptive.merge()

    adaptive_results = evaluator.evaluate(
        adaptive_clusters
    )

    evaluator.print_results(
        "Adaptive Coarsening",
        adaptive_results
    )