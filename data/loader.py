"""
loader.py

Utility functions for loading CVRPTW datasets.
"""

import pandas as pd


class DatasetLoader:
    """
    Loads customer datasets from CSV files.
    """

    REQUIRED_COLUMNS = [
        "id",
        "x",
        "y",
        "demand",
        "ready_time",
        "due_time",
        "service_time",
    ]

    @staticmethod
    def load(filepath):
        """
        Load a dataset from a CSV file.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        pandas.DataFrame
        """

        df = pd.read_csv(filepath)

        DatasetLoader.validate(df)

        print(f"Loaded {len(df)} customers.")

        return df

    @staticmethod
    def validate(df):
        """
        Validate dataset structure.
        """

        missing = [
            column
            for column in DatasetLoader.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset missing columns: {missing}"
            )

        if df.isnull().values.any():
            raise ValueError("Dataset contains missing values.")

        if (df["due_time"] <= df["ready_time"]).any():
            raise ValueError(
                "Each due_time must be greater than ready_time."
            )

        if (df["demand"] <= 0).any():
            raise ValueError(
                "Demand must be positive."
            )

        return True