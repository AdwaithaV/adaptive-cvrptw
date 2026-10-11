"""
adaptive_coarsening.py

Feasibility-aware adaptive graph coarsening for CVRPTW.

Supports:
- adaptive spatial/temporal weighting
- optional sequence-compatibility weighting
- demand-capacity constraint
- sequence-feasibility constraint
- membership preservation
- contraction history
- validation
"""

from __future__ import annotations

import copy
import time

import networkx as nx

from algorithms.contraction import (
    contraction_feasible,
    contract_customer_nodes,
)
from algorithms.similarity import SimilarityCalculator


class AdaptiveGraphCoarsener:

    def __init__(
        self,
        graph: nx.Graph,
        vehicle_capacity: float | None = None,
        require_sequence_feasibility: bool = True,
        sequence_weight: float = 0.0,
    ) -> None:

        self.original_graph = copy.deepcopy(graph)
        self.graph = copy.deepcopy(graph)

        self.vehicle_capacity = vehicle_capacity

        self.require_sequence_feasibility = bool(
            require_sequence_feasibility
        )

        # IMPORTANT:
        # Store the value passed by the experiment.
        self.sequence_weight = float(
            sequence_weight
        )

        if not 0.0 <= self.sequence_weight < 1.0:
            raise ValueError(
                "sequence_weight must be in [0, 1)."
            )

        self.history = []

        self.total_candidate_checks = 0
        self.total_infeasible_checks = 0
        self.total_successful_contractions = 0

        self.runtime_seconds = 0.0

    def best_feasible_edge(self):
        """
        Find the highest-scoring feasible edge under the
        current graph state.
        """

        similarity = SimilarityCalculator(
            self.graph
        )

        best_pair = None
        best_score = float("-inf")

        candidate_count = 0
        infeasible_count = 0

        for first_node, second_node in list(
            self.graph.edges()
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

            score = similarity.adaptive_similarity(
                first_node,
                second_node,
                sequence_weight=self.sequence_weight,
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
        """
        Contract the currently best feasible edge.
        """

        edge, score = (
            self.best_feasible_edge()
        )

        if edge is None:
            return False

        first_node, second_node = edge

        first_members = list(
            self.graph.nodes[
                first_node
            ].get(
                "members",
                [first_node],
            )
        )

        second_members = list(
            self.graph.nodes[
                second_node
            ].get(
                "members",
                [second_node],
            )
        )

        before_nodes = (
            self.graph.number_of_nodes()
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

        after_nodes = (
            self.graph.number_of_nodes()
        )

        if after_nodes != before_nodes - 1:
            raise RuntimeError(
                "A successful contraction must "
                "reduce node count by exactly one."
            )

        merged_members = list(
            self.graph.nodes[
                first_node
            ].get(
                "members",
                first_members + second_members,
            )
        )

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
                    merged_members,

                "cluster_size":
                    len(merged_members),

                "merged_demand":
                    float(
                        self.graph.nodes[
                            first_node
                        ]["demand"]
                    ),
            }
        )

        return True

    def run(
        self,
        target_ratio: float = 0.5,
        verbose: bool = True,
    ) -> nx.Graph:
        """
        Coarsen until target_ratio of the original node count
        remains, or until no feasible contraction exists.
        """

        if not 0.0 < target_ratio <= 1.0:
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

        if verbose:

            print(
                "=" * 60
            )

            print(
                "ADAPTIVE CVRPTW GRAPH COARSENING"
            )

            print(
                "=" * 60
            )

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
                "Sequence weight:",
                self.sequence_weight,
            )

            print()

        start = time.perf_counter()

        while (
            self.graph.number_of_nodes()
            > target_nodes
        ):

            before = (
                self.graph.number_of_nodes()
            )

            successful = (
                self.contract_best_edge()
            )

            if not successful:

                if verbose:
                    print(
                        "No more feasible "
                        "contractions available."
                    )

                break

            after = (
                self.graph.number_of_nodes()
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
                    f"{record['score']:.6f}"
                )

        self.runtime_seconds = (
            time.perf_counter()
            - start
        )

        return self.graph

    def validate(self) -> dict:
        """
        Validate basic invariants after coarsening.
        """

        original_member_list = []

        for node in self.original_graph.nodes:

            original_member_list.extend(
                self.original_graph.nodes[
                    node
                ].get(
                    "members",
                    [node],
                )
            )

        coarsened_member_list = []

        for node in self.graph.nodes:

            coarsened_member_list.extend(
                self.graph.nodes[
                    node
                ].get(
                    "members",
                    [node],
                )
            )

        original_members = set(
            original_member_list
        )

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

        if self.vehicle_capacity is None:

            capacity_valid = True

        else:

            capacity_valid = all(
                float(
                    self.graph.nodes[
                        node
                    ]["demand"]
                )
                <= float(
                    self.vehicle_capacity
                )
                + 1e-9

                for node in self.graph.nodes
            )

        time_windows_valid = all(
            float(
                self.graph.nodes[
                    node
                ]["ready_time"]
            )
            <= float(
                self.graph.nodes[
                    node
                ]["due_time"]
            )

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
                (
                    nx.is_connected(
                        self.graph
                    )
                    if self.graph.number_of_nodes()
                    > 0
                    else True
                ),

            "membership_preserved":
                original_members
                == coarsened_members,

            "duplicate_members":
                len(
                    coarsened_member_list
                )
                != len(
                    coarsened_members
                ),

            "demand_preserved":
                abs(
                    original_demand
                    - coarsened_demand
                )
                < 1e-9,

            "capacity_valid":
                capacity_valid,

            "time_windows_valid":
                time_windows_valid,

            "runtime_seconds":
                self.runtime_seconds,

            "sequence_weight":
                self.sequence_weight,
        }

    def summary(self):

        results = self.validate()

        print()
        print("=" * 60)
        print(
            "ADAPTIVE COARSENING SUMMARY"
        )
        print("=" * 60)

        for key, value in results.items():

            print(
                f"{key:28s}: {value}"
            )
