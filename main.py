"""
main.py

Entry point for the Adaptive Graph Coarsening
research prototype for CVRPTW.
"""

from experiment import Experiment


def main():
    """
    Configure and run the experiment.
    """

    experiment = Experiment(

        # Dataset parameters
        num_customers=100,

        # Graph parameters
        k_neighbors=5,

        # Target graph reduction
        reduction_ratio=0.50

    )

    results = experiment.run()

    print("\nExperiment completed successfully.")
    print("=" * 60)

    print("\nSummary")
    print("-" * 60)

    print(results["metrics"])


if __name__ == "__main__":
    main()