from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


FAMILY_FILE = Path("homberger_family_benchmark.csv")
RATIO_FILE = Path("homberger_reduction_ratio_benchmark.csv")

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def percent_improvement(fixed, adaptive, lower_is_better=True):

    if fixed == 0:
        return 0.0

    if lower_is_better:
        return ((fixed - adaptive) / fixed) * 100.0

    return ((adaptive - fixed) / fixed) * 100.0


def analyze_family_results():

    df = pd.read_csv(FAMILY_FILE)

    fixed = (
        df[df["method"] == "FIXED"]
        .set_index("family")
    )

    adaptive = (
        df[df["method"] == "ADAPTIVE"]
        .set_index("family")
    )

    rows = []

    for family in fixed.index:

        f = fixed.loc[family]
        a = adaptive.loc[family]

        rows.append(
            {
                "family": family,

                "fixed_spatial":
                    f["spatial_compactness"],

                "adaptive_spatial":
                    a["spatial_compactness"],

                "spatial_improvement_percent":
                    percent_improvement(
                        f["spatial_compactness"],
                        a["spatial_compactness"],
                        True,
                    ),

                "fixed_temporal":
                    f["temporal_dispersion"],

                "adaptive_temporal":
                    a["temporal_dispersion"],

                "temporal_improvement_percent":
                    percent_improvement(
                        f["temporal_dispersion"],
                        a["temporal_dispersion"],
                        True,
                    ),

                "fixed_sequence":
                    f["sequence_feasibility_rate"],

                "adaptive_sequence":
                    a["sequence_feasibility_rate"],

                "sequence_difference_pp":
                    (
                        a["sequence_feasibility_rate"]
                        -
                        f["sequence_feasibility_rate"]
                    ) * 100.0,

                "fixed_runtime":
                    f["runtime_seconds"],

                "adaptive_runtime":
                    a["runtime_seconds"],

                "runtime_change_percent":
                    percent_improvement(
                        f["runtime_seconds"],
                        a["runtime_seconds"],
                        False,
                    ),
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR / "family_comparison_summary.csv",
        index=False,
    )

    return df, summary


def analyze_ratio_results():

    df = pd.read_csv(RATIO_FILE)

    rows = []

    keys = (
        df[
            ["family", "target_ratio"]
        ]
        .drop_duplicates()
        .values
    )

    for family, ratio in keys:

        subset = df[
            (df["family"] == family)
            &
            (df["target_ratio"] == ratio)
        ]

        fixed = subset[
            subset["method"] == "FIXED"
        ].iloc[0]

        adaptive = subset[
            subset["method"] == "ADAPTIVE"
        ].iloc[0]

        reduction_percent = (
            1.0 - float(ratio)
        ) * 100.0

        rows.append(
            {
                "family": family,

                "target_ratio":
                    ratio,

                "reduction_percent":
                    reduction_percent,

                "final_nodes":
                    adaptive["final_nodes"],

                "fixed_spatial":
                    fixed["spatial_compactness"],

                "adaptive_spatial":
                    adaptive["spatial_compactness"],

                "spatial_improvement_percent":
                    percent_improvement(
                        fixed["spatial_compactness"],
                        adaptive["spatial_compactness"],
                        True,
                    ),

                "fixed_temporal":
                    fixed["temporal_dispersion"],

                "adaptive_temporal":
                    adaptive["temporal_dispersion"],

                "temporal_improvement_percent":
                    percent_improvement(
                        fixed["temporal_dispersion"],
                        adaptive["temporal_dispersion"],
                        True,
                    ),

                "fixed_sequence":
                    fixed[
                        "sequence_feasibility_rate"
                    ],

                "adaptive_sequence":
                    adaptive[
                        "sequence_feasibility_rate"
                    ],

                "sequence_difference_pp":
                    (
                        adaptive[
                            "sequence_feasibility_rate"
                        ]
                        -
                        fixed[
                            "sequence_feasibility_rate"
                        ]
                    ) * 100.0,

                "fixed_runtime":
                    fixed["runtime_seconds"],

                "adaptive_runtime":
                    adaptive["runtime_seconds"],
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR
        / "reduction_ratio_comparison_summary.csv",
        index=False,
    )

    return df, summary


def print_family_statistics(summary):

    spatial_wins = (
        summary[
            "spatial_improvement_percent"
        ] > 0
    ).sum()

    temporal_wins = (
        summary[
            "temporal_improvement_percent"
        ] > 0
    ).sum()

    total = len(summary)

    print()
    print("=" * 80)
    print("MULTI-FAMILY SUMMARY")
    print("=" * 80)

    print(
        "Families tested:",
        total
    )

    print(
        "Adaptive spatial wins:",
        f"{spatial_wins}/{total}"
    )

    print(
        "Adaptive temporal wins:",
        f"{temporal_wins}/{total}"
    )

    print(
        "Mean spatial improvement (%):",
        summary[
            "spatial_improvement_percent"
        ].mean()
    )

    print(
        "Median spatial improvement (%):",
        summary[
            "spatial_improvement_percent"
        ].median()
    )

    print(
        "Mean temporal improvement (%):",
        summary[
            "temporal_improvement_percent"
        ].mean()
    )

    print(
        "Median temporal improvement (%):",
        summary[
            "temporal_improvement_percent"
        ].median()
    )

    print(
        "Mean sequence difference (percentage points):",
        summary[
            "sequence_difference_pp"
        ].mean()
    )


def print_ratio_statistics(summary):

    print()
    print("=" * 80)
    print("REDUCTION-RATIO SUMMARY")
    print("=" * 80)

    print(
        "Comparisons:",
        len(summary)
    )

    print(
        "Spatial wins:",
        (
            summary[
                "spatial_improvement_percent"
            ] > 0
        ).sum(),
        "/",
        len(summary),
    )

    print(
        "Temporal wins:",
        (
            summary[
                "temporal_improvement_percent"
            ] > 0
        ).sum(),
        "/",
        len(summary),
    )

    for reduction in sorted(
        summary["reduction_percent"].unique()
    ):

        subset = summary[
            summary["reduction_percent"]
            == reduction
        ]

        print()
        print(
            f"{reduction:.0f}% REDUCTION"
        )

        print(
            "Mean spatial improvement (%):",
            subset[
                "spatial_improvement_percent"
            ].mean()
        )

        print(
            "Mean temporal improvement (%):",
            subset[
                "temporal_improvement_percent"
            ].mean()
        )

        print(
            "Mean sequence difference (pp):",
            subset[
                "sequence_difference_pp"
            ].mean()
        )


def plot_family_spatial(summary):

    families = summary["family"]

    x = list(range(len(families)))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        [i - width / 2 for i in x],
        summary["fixed_spatial"],
        width,
        label="Fixed",
    )

    ax.bar(
        [i + width / 2 for i in x],
        summary["adaptive_spatial"],
        width,
        label="Adaptive",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(families)

    ax.set_xlabel(
        "Homberger Family"
    )

    ax.set_ylabel(
        "Mean Spatial Compactness"
    )

    ax.set_title(
        "Spatial Compactness: Fixed vs Adaptive"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "family_spatial_compactness.png",
        dpi=300,
    )

    plt.close(fig)


def plot_family_temporal(summary):

    families = summary["family"]

    x = list(range(len(families)))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        [i - width / 2 for i in x],
        summary["fixed_temporal"],
        width,
        label="Fixed",
    )

    ax.bar(
        [i + width / 2 for i in x],
        summary["adaptive_temporal"],
        width,
        label="Adaptive",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(families)

    ax.set_xlabel(
        "Homberger Family"
    )

    ax.set_ylabel(
        "Mean Temporal Dispersion"
    )

    ax.set_title(
        "Temporal Dispersion: Fixed vs Adaptive"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "family_temporal_dispersion.png",
        dpi=300,
    )

    plt.close(fig)


def plot_family_sequence(summary):

    families = summary["family"]

    x = list(range(len(families)))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        [i - width / 2 for i in x],
        summary["fixed_sequence"],
        width,
        label="Fixed",
    )

    ax.bar(
        [i + width / 2 for i in x],
        summary["adaptive_sequence"],
        width,
        label="Adaptive",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(families)

    ax.set_xlabel(
        "Homberger Family"
    )

    ax.set_ylabel(
        "Sequence Feasibility Rate"
    )

    ax.set_title(
        "Sequence Feasibility: Fixed vs Adaptive"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "family_sequence_feasibility.png",
        dpi=300,
    )

    plt.close(fig)


def plot_ratio_temporal(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for family in summary[
        "family"
    ].unique():

        subset = (
            summary[
                summary["family"]
                == family
            ]
            .sort_values(
                "reduction_percent"
            )
        )

        ax.plot(
            subset[
                "reduction_percent"
            ],
            subset[
                "temporal_improvement_percent"
            ],
            marker="o",
            label=family,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Reduction (%)"
    )

    ax.set_ylabel(
        "Adaptive Temporal Improvement (%)"
    )

    ax.set_title(
        "Temporal Improvement Across Reduction Levels"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "reduction_temporal_improvement.png",
        dpi=300,
    )

    plt.close(fig)


def plot_ratio_spatial(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for family in summary[
        "family"
    ].unique():

        subset = (
            summary[
                summary["family"]
                == family
            ]
            .sort_values(
                "reduction_percent"
            )
        )

        ax.plot(
            subset[
                "reduction_percent"
            ],
            subset[
                "spatial_improvement_percent"
            ],
            marker="o",
            label=family,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Reduction (%)"
    )

    ax.set_ylabel(
        "Adaptive Spatial Improvement (%)"
    )

    ax.set_title(
        "Spatial Improvement Across Reduction Levels"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "reduction_spatial_improvement.png",
        dpi=300,
    )

    plt.close(fig)


def main():

    if not FAMILY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FAMILY_FILE}"
        )

    if not RATIO_FILE.exists():
        raise FileNotFoundError(
            f"Missing {RATIO_FILE}"
        )

    family_raw, family_summary = (
        analyze_family_results()
    )

    ratio_raw, ratio_summary = (
        analyze_ratio_results()
    )

    print_family_statistics(
        family_summary
    )

    print_ratio_statistics(
        ratio_summary
    )

    plot_family_spatial(
        family_summary
    )

    plot_family_temporal(
        family_summary
    )

    plot_family_sequence(
        family_summary
    )

    plot_ratio_temporal(
        ratio_summary
    )

    plot_ratio_spatial(
        ratio_summary
    )

    print()
    print("=" * 80)
    print(
        "PHASE-1 ANALYSIS COMPLETE"
    )
    print("=" * 80)

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print()
    print(
        "Created:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            " -",
            path
        )


if __name__ == "__main__":
    main()
