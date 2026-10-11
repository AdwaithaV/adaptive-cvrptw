"""
fixed_coarsening.py

Fixed-weight feasibility-aware graph coarsening for CVRPTW.

This baseline uses the same:
- graph
- capacity constraints
- sequential feasibility
- contraction logic
- stopping rule

as the adaptive method.

The only main difference is that spatial and temporal
weights are fixed instead of adaptive.
"""

from __future__ import annotations

import copy
import time

import networkx as nx

from algorithms.contraction import (
    contract_customer_nodes,
    contraction_feasible,
)
from algorithms.similarity import SimilarityCalculator


class FixedGraphCoarsener:

    def __init__(
        self,
        graph: nx.Graph,
        vehicle_capacity: float | None = None,
        require_sequence_feasibility: bool = True,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.10,
    ) -> None:

        self.original_graph = copy.deepcopy(graph)
        self.graph = copy.deepcopy(graph)

        self.vehicle_capacity = vehicle_capacity

        self.require_sequence_feasibility = (
            require_sequence_feasibility
        )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError(
                "Similarity weights must be non-negative."
            )

        self.history = []

        self.total_candidate_checks = 0
        self.total_infeasible_checks = 0
        self.total_successful_contractions = 0

        self.runtime_seconds = 0.0

    def fixed_similarity(
        self,
        similarity: SimilarityCalculator,
        first_node,
        second_node,
    ) -> float:

        spatial = similarity.spatial_similarity(
            first_node,
            second_node,
        )

        temporal = (
            0.5
            * similarity.temporal_similarity(
                first_node,
                second_node,
            )
            +
            0.5
            * similarity.time_overlap(
                first_node,
                second_node,
            )
        )

        demand = similarity.demand_similarity(
            first_node,
            second_node,
        )

        score = (
            self.alpha * spatial
            + self.beta * temporal
            + self.gamma * demand
        )

        return float(score)

    def best_feasible_edge(self):

        similarity = SimilarityCalculator(
            self.graph
        )

        best_pair = None
        best_score = float("-inf")

        candidate_count = 0
        infeasible_count = 0

        for first_node, second_node in list(
            self.graph.edges
        ):

            candidate_count += 1

            feasible, _ = contraction_feasible(
                graph=self.graph,
                first_node=first_node,
                second_node=second_node,
                vehicle_capacity=self.vehicle_capacity,
                require_sequence_feasibility=(
                    self.require_sequence_feasibility
                ),
            )

            if not feasible:
                infeasible_count += 1
                continue

            score = self.fixed_similarity(
                similarity,
                first_node,
                second_node,
            )

            if score > best_score:
                best_score = score
                best_pair = (
                    first_node,
                    second_node,
                )

        self.total_candidate_checks += (
            candidate_count
        )

        self.total_infeasible_checks += (
            infeasible_count
        )

        return best_pair, best_score

    def contract_best_edge(self):

        edge, score = (
            self.best_feasible_edge()
        )

        if edge is None:
            return False

        first_node, second_node = edge

        first_members = list(
            self.graph.nodes[
                first_node
            ]["members"]
        )

        second_members = list(
            self.graph.nodes[
                second_node
            ]["members"]
        )

        self.graph = contract_customer_nodes(
            graph=self.graph,
            first_node=first_node,
            second_node=second_node,
            vehicle_capacity=self.vehicle_capacity,
            require_sequence_feasibility=(
                self.require_sequence_feasibility
            ),
        )

        merged = self.graph.nodes[first_node]

        self.total_successful_contractions += 1

        self.history.append(
            {
                "iteration":
                    self.total_successful_contractions,

                "first_node":
                    first_node,

                "second_node":
                    second_node,

                "score":
                    float(score),

                "first_members":
                    first_members,

                "second_members":
                    second_members,

                "merged_members":
                    list(
                        merged["members"]
                    ),

                "merged_demand":
                    float(
                        merged["demand"]
                    ),

                "cluster_size":
                    int(
                        merged["cluster_size"]
                    ),
            }
        )

        return True

    def run(
        self,
        target_ratio: float = 0.5,
        verbose: bool = True,
    ) -> nx.Graph:

        if not 0 < target_ratio <= 1:
            raise ValueError(
                "target_ratio must be in (0, 1]."
            )

        original_nodes = (
            self.original_graph
            .number_of_nodes()
        )

        target_nodes = max(
            1,
            int(
                round(
                    original_nodes
                    * target_ratio
                )
            ),
        )

        start_time = time.perf_counter()

        if verbose:

            print("=" * 60)
            print(
                "FIXED CVRPTW GRAPH COARSENING"
            )
            print("=" * 60)

            print(
                "Original nodes:",
                original_nodes,
            )

            print(
                "Target nodes:",
                target_nodes,
            )

            print(
                "Vehicle capacity:",
                self.vehicle_capacity,
            )

            print(
                "Alpha:",
                self.alpha,
            )

            print(
                "Beta:",
                self.beta,
            )

            print(
                "Gamma:",
                self.gamma,
            )

            print()

        while (
            self.graph.number_of_nodes()
            > target_nodes
        ):

            before = (
                self.graph.number_of_nodes()
            )

            success = (
                self.contract_best_edge()
            )

            if not success:

                if verbose:
                    print(
                        "\nNo additional feasible "
                        "contractions available."
                    )

                break

            after = (
                self.graph.number_of_nodes()
            )

            if after != before - 1:
                raise RuntimeError(
                    "Contraction must reduce "
                    "the node count by exactly one."
                )

            if verbose:

                record = self.history[-1]

                print(
                    f"Iteration "
                    f"{record['iteration']:3d}: "
                    f"{before:3d} -> "
                    f"{after:3d} | "
                    f"edge=("
                    f"{record['first_node']}, "
                    f"{record['second_node']}) | "
                    f"score="
                    f"{record['score']:.6f} | "
                    f"cluster="
                    f"{record['cluster_size']}"
                )

        self.runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        return self.graph

    def validate(self) -> dict:

        original_members = {
            member
            for node in self.original_graph.nodes
            for member in (
                self.original_graph.nodes[
                    node
                ]["members"]
            )
        }

        coarsened_member_list = [
            member
            for node in self.graph.nodes
            for member in (
                self.graph.nodes[
                    node
                ]["members"]
            )
        ]

        coarsened_members = set(
            coarsened_member_list
        )

        original_demand = sum(
            float(
                self.original_graph.nodes[
                    node
                ]["demand"]
            )
            for node in self.original_graph.nodes
        )

        coarsened_demand = sum(
            float(
                self.graph.nodes[
                    node
                ]["demand"]
            )
            for node in self.graph.nodes
        )

        capacity_valid = True

        if self.vehicle_capacity is not None:
            capacity_valid = all(
                float(
                    self.graph.nodes[
                        node
                    ]["demand"]
                )
                <= self.vehicle_capacity
                for node in self.graph.nodes
            )

        return {
            "original_nodes":
                self.original_graph
                .number_of_nodes(),

            "coarsened_nodes":
                self.graph
                .number_of_nodes(),

            "successful_contractions":
                self.total_successful_contractions,

            "candidate_checks":
                self.total_candidate_checks,

            "infeasible_checks":
                self.total_infeasible_checks,

            "connected":
                nx.is_connected(
                    self.graph
                ),

            "membership_preserved":
                original_members
                == coarsened_members,

            "duplicate_members":
                len(coarsened_member_list)
                != len(coarsened_members),

            "demand_preserved":
                abs(
                    original_demand
                    - coarsened_demand
                )
                < 1e-9,

            "capacity_valid":
                capacity_valid,

            "runtime_seconds":
                self.runtime_seconds,
        }

    def summary(self):

        results = self.validate()

        print()
        print("=" * 60)
        print(
            "FIXED COARSENING SUMMARY"
        )
        print("=" * 60)

        for key, value in results.items():
            print(
                f"{key:30s}: {value}"
            )
