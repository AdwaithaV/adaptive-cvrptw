"""
contraction.py

Feasibility-aware contraction utilities for CVRPTW graph coarsening.

Temporal feasibility is based on whether two meta-nodes can be
served sequentially in at least one direction:

    A -> B
or
    B -> A

All original member-level CVRPTW information is retained inside
the resulting meta-node for later inflation/refinement.
"""

from __future__ import annotations

import copy
import math
from typing import Hashable

import networkx as nx


def euclidean_distance(
    first_position,
    second_position,
) -> float:
    """Euclidean distance between two coordinates."""

    return math.sqrt(
        (first_position[0] - second_position[0]) ** 2
        + (first_position[1] - second_position[1]) ** 2
    )


def _member_information(node_id, data):
    """
    Obtain original member-level information.

    For an uncontracted customer, create this information
    from its normal node attributes.
    """

    members = list(
        data.get(
            "members",
            [int(node_id)],
        )
    )

    member_windows = dict(
        data.get(
            "member_windows",
            {
                int(member): (
                    float(data["ready_time"]),
                    float(data["due_time"]),
                )
                for member in members
            },
        )
    )

    member_demands = dict(
        data.get(
            "member_demands",
            {
                int(member): float(data["demand"])
                for member in members
            },
        )
    )

    member_service_times = dict(
        data.get(
            "member_service_times",
            {
                int(member): float(data["service_time"])
                for member in members
            },
        )
    )

    member_positions = dict(
        data.get(
            "member_positions",
            {
                int(member): tuple(data["position"])
                for member in members
            },
        )
    )

    return {
        "members": members,
        "member_windows": member_windows,
        "member_demands": member_demands,
        "member_service_times": member_service_times,
        "member_positions": member_positions,
    }


def sequence_feasible(
    graph: nx.Graph,
    first_node: Hashable,
    second_node: Hashable,
) -> tuple[bool, str]:
    """
    Check approximate sequential temporal feasibility.

    Tests:

        first_node -> second_node

    and

        second_node -> first_node

    using representative node time windows, service times,
    and Euclidean travel time.

    This is a coarsening heuristic, not a proof that every
    internal ordering of a large meta-node is route-feasible.
    """

    first = graph.nodes[first_node]
    second = graph.nodes[second_node]

    travel_time = euclidean_distance(
        first["position"],
        second["position"],
    )

    def can_follow(a, b):
        earliest_start_a = float(
            a["ready_time"]
        )

        finish_a = (
            earliest_start_a
            + float(a["service_time"])
        )

        arrival_b = (
            finish_a
            + travel_time
        )

        start_b = max(
            arrival_b,
            float(b["ready_time"]),
        )

        return start_b <= float(
            b["due_time"]
        )

    forward = can_follow(
        first,
        second,
    )

    backward = can_follow(
        second,
        first,
    )

    if forward and backward:
        return True, "Both service orders are feasible."

    if forward:
        return (
            True,
            f"{first_node} -> {second_node} is feasible.",
        )

    if backward:
        return (
            True,
            f"{second_node} -> {first_node} is feasible.",
        )

    return (
        False,
        "Neither service ordering is temporally feasible.",
    )


def contraction_feasible(
    graph: nx.Graph,
    first_node: Hashable,
    second_node: Hashable,
    vehicle_capacity: float | None = None,
    require_sequence_feasibility: bool = True,
) -> tuple[bool, str]:
    """
    Determine whether two neighboring meta-nodes may be merged.
    """

    if first_node not in graph:
        return False, f"Node {first_node} does not exist."

    if second_node not in graph:
        return False, f"Node {second_node} does not exist."

    if first_node == second_node:
        return False, "Cannot contract a node with itself."

    if not graph.has_edge(
        first_node,
        second_node,
    ):
        return (
            False,
            "Candidate nodes must share an edge.",
        )

    first = graph.nodes[first_node]
    second = graph.nodes[second_node]

    combined_demand = (
        float(first["demand"])
        + float(second["demand"])
    )

    if (
        vehicle_capacity is not None
        and combined_demand
        > float(vehicle_capacity)
    ):
        return (
            False,
            (
                f"Combined demand {combined_demand} "
                f"exceeds vehicle capacity "
                f"{vehicle_capacity}."
            ),
        )

    if require_sequence_feasibility:
        feasible, reason = sequence_feasible(
            graph,
            first_node,
            second_node,
        )

        if not feasible:
            return False, reason

    return True, "Feasible contraction."


def contract_customer_nodes(
    graph: nx.Graph,
    first_node: Hashable,
    second_node: Hashable,
    vehicle_capacity: float | None = None,
    require_sequence_feasibility: bool = True,
) -> nx.Graph:
    """
    Merge second_node into first_node.

    The input graph is not modified.
    """

    feasible, reason = contraction_feasible(
        graph=graph,
        first_node=first_node,
        second_node=second_node,
        vehicle_capacity=vehicle_capacity,
        require_sequence_feasibility=(
            require_sequence_feasibility
        ),
    )

    if not feasible:
        raise ValueError(
            f"Cannot contract "
            f"{first_node}, {second_node}: "
            f"{reason}"
        )

    new_graph = copy.deepcopy(graph)

    first = dict(
        new_graph.nodes[first_node]
    )

    second = dict(
        new_graph.nodes[second_node]
    )

    first_info = _member_information(
        first_node,
        first,
    )

    second_info = _member_information(
        second_node,
        second,
    )

    members = sorted(
        set(
            first_info["members"]
            + second_info["members"]
        )
    )

    member_windows = {
        **first_info["member_windows"],
        **second_info["member_windows"],
    }

    member_demands = {
        **first_info["member_demands"],
        **second_info["member_demands"],
    }

    member_service_times = {
        **first_info["member_service_times"],
        **second_info["member_service_times"],
    }

    member_positions = {
        **first_info["member_positions"],
        **second_info["member_positions"],
    }

    #
    # Representative position.
    #
    # Weight each previous meta-node by the number
    # of original customers it contains.
    #

    first_weight = len(
        first_info["members"]
    )

    second_weight = len(
        second_info["members"]
    )

    total_weight = (
        first_weight
        + second_weight
    )

    merged_x = (
        float(first["x"]) * first_weight
        + float(second["x"]) * second_weight
    ) / total_weight

    merged_y = (
        float(first["y"]) * first_weight
        + float(second["y"]) * second_weight
    ) / total_weight

    #
    # Representative time window.
    #
    # Unlike the old strict intersection rule,
    # preserve the overall temporal span of members.
    #
    # Exact individual windows remain stored in
    # member_windows.
    #

    merged_ready_time = min(
        ready
        for ready, _ in member_windows.values()
    )

    merged_due_time = max(
        due
        for _, due in member_windows.values()
    )

    merged_demand = sum(
        member_demands.values()
    )

    merged_service_time = sum(
        member_service_times.values()
    )

    merged_data = {
        **first,

        "customer_id": first_node,

        "x": merged_x,
        "y": merged_y,
        "position": (
            merged_x,
            merged_y,
        ),

        "demand": float(
            merged_demand
        ),

        "ready_time": float(
            merged_ready_time
        ),

        "due_time": float(
            merged_due_time
        ),

        "service_time": float(
            merged_service_time
        ),

        "time_window_width": float(
            merged_due_time
            - merged_ready_time
        ),

        "is_depot": False,

        "members": members,

        "cluster_size": len(
            members
        ),

        "member_windows":
            member_windows,

        "member_demands":
            member_demands,

        "member_service_times":
            member_service_times,

        "member_positions":
            member_positions,

        "contracted_from": (
            first_node,
            second_node,
        ),
    }

    first_neighbors = set(
        new_graph.neighbors(
            first_node
        )
    )

    second_neighbors = set(
        new_graph.neighbors(
            second_node
        )
    )

    merged_neighbors = (
        first_neighbors
        | second_neighbors
    ) - {
        first_node,
        second_node,
    }

    new_graph.remove_node(
        second_node
    )

    new_graph.nodes[
        first_node
    ].clear()

    new_graph.nodes[
        first_node
    ].update(
        merged_data
    )

    #
    # Recompute distances from the new centroid
    # to every affected neighbor.
    #

    for neighbor in merged_neighbors:

        neighbor_position = (
            new_graph.nodes[
                neighbor
            ]["position"]
        )

        distance = euclidean_distance(
            merged_data["position"],
            neighbor_position,
        )

        new_graph.add_edge(
            first_node,
            neighbor,
            distance=distance,
        )

    return new_graph
