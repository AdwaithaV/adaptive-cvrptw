"""
experiment.py

Runs a complete graph coarsening experiment for CVRPTW.

Pipeline
--------
Dataset
    ↓
Graph Construction
    ↓
Graph Feature Extraction
    ↓
Adaptive Weight Computation
    ↓
Similarity Computation
    ↓
Fixed Graph Coarsening
Adaptive Graph Coarsening
    ↓
Evaluation
    ↓
Visualization
"""

import pandas as pd

from data.generator import CVRPTWDatasetGenerator
from graph.build_graph import GraphBuilder
from graph.graph_utils import GraphFeatureExtractor

from algorithms.adaptive_weights import AdaptiveWeightCalculator
from algorithms.similarity import SimilarityCalculator
from algorithms.fixed_coarsening import FixedGraphCoarsener
from algorithms.adaptive_coarsening import AdaptiveGraphCoarsener

from metrics.evaluation import GraphEvaluator

from visualization.plot_graph import GraphVisualizer
from visualization.compare import ComparisonVisualizer


class Experiment:

    def __init__(
        self,
        num_customers=100,
        k_neighbors=5,
        reduction_ratio=0.5
    ):

        self.num_customers = num_customers
        self.k_neighbors = k_neighbors
        self.reduction_ratio = reduction_ratio

    def run(self):

        print("=" * 60)
        print("CVRPTW GRAPH COARSENING EXPERIMENT")
        print("=" * 60)

        ####################################################
        # 1. Generate Dataset
        ####################################################

        print("\nGenerating dataset...")

        generator = CVRPTWDatasetGenerator(
            num_customers=self.num_customers
        )

        df = generator.generate()

        print(df.head())

        ####################################################
        # 2. Build Graph
        ####################################################

        print("\nBuilding graph...")

        builder = GraphBuilder(k=self.k_neighbors)

        graph = builder.build(df)

        ####################################################
        # 3. Compute Graph Features
        ####################################################

        print("\nExtracting graph features...")

        extractor = GraphFeatureExtractor(graph)

        extractor.compute_all_features()

        ####################################################
        # 4. Adaptive Weights
        ####################################################

        print("\nComputing adaptive weights...")

        weight_calculator = AdaptiveWeightCalculator(graph)

        weight_calculator.compute()

        ####################################################
        # 5. Similarities
        ####################################################

        print("\nComputing similarities...")

        similarity = SimilarityCalculator(graph)

        similarity.compute_all()

        ####################################################
        # 6. Fixed Coarsening
        ####################################################

        print("\nRunning fixed coarsening...")

        fixed = FixedGraphCoarsener(graph)

        fixed_graph = fixed.coarsen(
            reduction_ratio=self.reduction_ratio
        )

        ####################################################
        # 7. Adaptive Coarsening
        ####################################################

        print("\nRunning adaptive coarsening...")

        adaptive = AdaptiveGraphCoarsener(graph)

        adaptive_graph = adaptive.coarsen(
            reduction_ratio=self.reduction_ratio
        )

        ####################################################
        # 8. Evaluation
        ####################################################

        print("\nEvaluating graphs...")

        evaluator = GraphEvaluator()

        fixed_metrics = evaluator.evaluate(
            original_graph=graph,
            coarsened_graph=fixed_graph,
            name="Fixed"
        )

        adaptive_metrics = evaluator.evaluate(
            original_graph=graph,
            coarsened_graph=adaptive_graph,
            name="Adaptive"
        )

        metrics = pd.DataFrame([
            fixed_metrics,
            adaptive_metrics
        ])

        print("\nResults\n")
        print(metrics)

        ####################################################
        # 9. Visualizations
        ####################################################

        print("\nGenerating figures...")

        GraphVisualizer.compare_graphs(
            graph,
            fixed_graph,
            adaptive_graph
        )

        GraphVisualizer.draw_alpha(graph)

        GraphVisualizer.draw_beta(graph)

        GraphVisualizer.draw_similarity(graph)

        ####################################################
        # 10. Comparison Plot
        ####################################################

        comparison = pd.DataFrame({

            "Metric": [
                "Reduction",
                "Density",
                "Degree",
                "Clustering"
            ],

            "Fixed": [

                fixed_metrics["reduction_ratio"],
                fixed_metrics["density"],
                fixed_metrics["average_degree"],
                fixed_metrics["clustering"]

            ],

            "Adaptive": [

                adaptive_metrics["reduction_ratio"],
                adaptive_metrics["density"],
                adaptive_metrics["average_degree"],
                adaptive_metrics["clustering"]

            ]

        })

        ComparisonVisualizer.plot_metrics(comparison)

        print("\nExperiment Complete.")

        return {
            "graph": graph,
            "fixed_graph": fixed_graph,
            "adaptive_graph": adaptive_graph,
            "metrics": metrics
        }


if __name__ == "__main__":

    experiment = Experiment(
        num_customers=100,
        k_neighbors=6,
        reduction_ratio=0.50
    )

    experiment.run()