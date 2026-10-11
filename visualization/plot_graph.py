"""
plot_graph.py

Visualization utilities for CVRPTW graph coarsening.
"""

import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:

    @staticmethod
    def draw_graph(
        graph,
        title="Graph",
        node_color="skyblue",
        edge_color="gray",
        node_size=120,
        with_labels=False,
        figsize=(10, 8)
    ):
        """
        Draw a NetworkX graph using customer coordinates.
        """

        pos = {
            node: (
                graph.nodes[node]["x"],
                graph.nodes[node]["y"]
            )
            for node in graph.nodes()
        }

        plt.figure(figsize=figsize)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_color,
            node_size=node_size
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=edge_color,
            alpha=0.6
        )

        if with_labels:
            nx.draw_networkx_labels(
                graph,
                pos,
                font_size=8
            )

        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_alpha(graph):
        """
        Color nodes according to adaptive alpha.
        """

        pos = {
            n: (
                graph.nodes[n]["x"],
                graph.nodes[n]["y"]
            )
            for n in graph.nodes()
        }

        colors = [
            graph.nodes[n].get("alpha", 0.5)
            for n in graph.nodes()
        ]

        plt.figure(figsize=(10, 8))

        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=colors,
            cmap=plt.cm.Blues,
            node_size=120
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.4
        )

        plt.colorbar(nodes, label="Adaptive α")

        plt.title("Spatial Adaptive Weight (α)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_beta(graph):
        """
        Color nodes according to adaptive beta.
        """

        pos = {
            n: (
                graph.nodes[n]["x"],
                graph.nodes[n]["y"]
            )
            for n in graph.nodes()
        }

        colors = [
            graph.nodes[n].get("beta", 0.5)
            for n in graph.nodes()
        ]

        plt.figure(figsize=(10, 8))

        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=colors,
            cmap=plt.cm.Reds,
            node_size=120
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.4
        )

        plt.colorbar(nodes, label="Adaptive β")

        plt.title("Temporal Adaptive Weight (β)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_similarity(graph):
        """
        Draw graph with edges colored by similarity.
        """

        pos = {
            node: (
                graph.nodes[node]["x"],
                graph.nodes[node]["y"]
            )
            for node in graph.nodes()
        }

        similarities = [
            graph[u][v].get("similarity", 0)
            for u, v in graph.edges()
        ]

        plt.figure(figsize=(10, 8))

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=120,
            node_color="orange"
        )

        edges = nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=similarities,
            edge_cmap=plt.cm.viridis,
            width=2
        )

        plt.colorbar(edges, label="Edge Similarity")

        plt.title("Edge Similarity Heatmap")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def compare_graphs(original, fixed, adaptive):
        """
        Display original, fixed, and adaptive graphs side by side.
        """

        graphs = [
            (original, "Original Graph"),
            (fixed, "Fixed Coarsening"),
            (adaptive, "Adaptive Coarsening")
        ]

        plt.figure(figsize=(18, 6))

        for i, (graph, title) in enumerate(graphs):

            plt.subplot(1, 3, i + 1)

            pos = {
                n: (
                    graph.nodes[n]["x"],
                    graph.nodes[n]["y"]
                )
                for n in graph.nodes()
            }

            nx.draw(
                graph,
                pos,
                node_size=80,
                with_labels=False
            )

            plt.title(title)

        plt.tight_layout()

        plt.show()