"""
generator.py

Generate synthetic CVRPTW customer datasets.

Each customer has:
- Customer ID
- Coordinates (x, y)
- Demand
- Ready Time
- Due Time
- Service Time
"""

import numpy as np
import pandas as pd


class CVRPTWDatasetGenerator:
    """
    Generates synthetic CVRPTW customer datasets.
    """

    def __init__(
        self,
        num_customers=100,
        x_range=(0, 100),
        y_range=(0, 100),
        demand_range=(1, 20),
        ready_time_range=(0, 500),
        window_width_range=(30, 120),
        service_time_range=(5, 20),
        random_seed=42,
    ):

        self.num_customers = num_customers
        self.x_range = x_range
        self.y_range = y_range
        self.demand_range = demand_range
        self.ready_time_range = ready_time_range
        self.window_width_range = window_width_range
        self.service_time_range = service_time_range

        np.random.seed(random_seed)

    def generate(self):
        """
        Generate a synthetic customer dataset.

        Returns
        -------
        pandas.DataFrame
        """

        ids = np.arange(self.num_customers)

        x = np.random.uniform(
            self.x_range[0],
            self.x_range[1],
            self.num_customers,
        )

        y = np.random.uniform(
            self.y_range[0],
            self.y_range[1],
            self.num_customers,
        )

        demand = np.random.randint(
            self.demand_range[0],
            self.demand_range[1] + 1,
            self.num_customers,
        )

        ready_time = np.random.randint(
            self.ready_time_range[0],
            self.ready_time_range[1],
            self.num_customers,
        )

        width = np.random.randint(
            self.window_width_range[0],
            self.window_width_range[1],
            self.num_customers,
        )

        due_time = ready_time + width

        service_time = np.random.randint(
            self.service_time_range[0],
            self.service_time_range[1] + 1,
            self.num_customers,
        )

        df = pd.DataFrame({
            "id": ids,
            "x": x,
            "y": y,
            "demand": demand,
            "ready_time": ready_time,
            "due_time": due_time,
            "service_time": service_time,
        })

        return df

    def save(self, filename):
        """
        Generate and save dataset to CSV.
        """

        df = self.generate()
        df.to_csv(filename, index=False)

        print(f"Dataset saved to {filename}")
        print(df.head())