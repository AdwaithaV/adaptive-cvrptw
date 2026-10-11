from pathlib import Path
from typing import Any

import pandas as pd


CUSTOMER_COLUMNS = [
    "customer_id",
    "x",
    "y",
    "demand",
    "ready_time",
    "due_time",
    "service_time",
]


def load_homberger_instance(file_path: str | Path) -> dict[str, Any]:
    """
    Load one Gehring-Homberger VRPTW benchmark instance.

    Parameters
    ----------
    file_path:
        Path to a Homberger .TXT instance file.

    Returns
    -------
    dict
        Contains:
        - instance_name
        - vehicle_count
        - vehicle_capacity
        - depot
        - customers
        - all_nodes
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]

    if not lines:
        raise ValueError(f"Dataset file is empty: {path}")

    instance_name = lines[0]

    # Locate the line containing vehicle count and capacity.
    vehicle_header_index = None

    for index, line in enumerate(lines):
        if line.startswith("NUMBER") and "CAPACITY" in line:
            vehicle_header_index = index
            break

    if vehicle_header_index is None:
        raise ValueError("Could not find vehicle information.")

    vehicle_values = lines[vehicle_header_index + 1].split()

    if len(vehicle_values) < 2:
        raise ValueError("Invalid vehicle information line.")

    vehicle_count = int(vehicle_values[0])
    vehicle_capacity = int(vehicle_values[1])

    # Locate the customer table header.
    customer_header_index = None

    for index, line in enumerate(lines):
        if line.startswith("CUST NO."):
            customer_header_index = index
            break

    if customer_header_index is None:
        raise ValueError("Could not find customer table.")

    records = []

    for line in lines[customer_header_index + 1 :]:
        if not line:
            continue

        values = line.split()

        # Every customer row must contain exactly seven values.
        if len(values) != 7:
            continue

        try:
            record = {
                "customer_id": int(values[0]),
                "x": float(values[1]),
                "y": float(values[2]),
                "demand": float(values[3]),
                "ready_time": float(values[4]),
                "due_time": float(values[5]),
                "service_time": float(values[6]),
            }
        except ValueError:
            continue

        records.append(record)

    if not records:
        raise ValueError("No customer records were found.")

    all_nodes = pd.DataFrame(records, columns=CUSTOMER_COLUMNS)

    depot_rows = all_nodes[all_nodes["customer_id"] == 0]

    if depot_rows.empty:
        raise ValueError("Depot with customer_id 0 was not found.")

    depot = depot_rows.iloc[0].to_dict()

    customers = (
        all_nodes[all_nodes["customer_id"] != 0]
        .reset_index(drop=True)
    )

    return {
        "instance_name": instance_name,
        "vehicle_count": vehicle_count,
        "vehicle_capacity": vehicle_capacity,
        "depot": depot,
        "customers": customers,
        "all_nodes": all_nodes,
    }


if __name__ == "__main__":
    dataset_path = (
        Path(__file__).resolve().parent
        / "datasets"
        / "homberger"
        / "C1_10_1.TXT"
    )

    instance = load_homberger_instance(dataset_path)

    print("Instance name:", instance["instance_name"])
    print("Maximum vehicles:", instance["vehicle_count"])
    print("Vehicle capacity:", instance["vehicle_capacity"])
    print("Number of customers:", len(instance["customers"]))
    print("Depot:")
    print(instance["depot"])

    print("\nFirst five customers:")
    print(instance["customers"].head())