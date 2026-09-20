import os
from datetime import datetime
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from src.datatypes import SpectralClusteringConfig, SpectralAnalysisResult


def get_output_path(filename: str, subfolder: str | None = None) -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    output_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "output"
    )
    output_path = os.path.join(output_folder, date)
    if subfolder is not None:
        output_path = os.path.join(output_path, subfolder)
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def plot_trajectory_paths(
    trajectories: np.ndarray,
    subfolder: Optional[str] = None,
    plot_title: Optional[str] = None,
) -> None:
    """
    Plots the paths of multiple trajectories in 2D space.
    """
    if plot_title:
        plt.title(plot_title)
    for trajectory in trajectories:
        x_coordinates = trajectory[:, 0]
        y_coordinates = trajectory[:, 1]
        plt.plot(x_coordinates, y_coordinates)
    marker_size = 5
    for trajectory in trajectories:
        x_coordinates = trajectory[:, 0]
        y_coordinates = trajectory[:, 1]
        initial_position = (x_coordinates[0], y_coordinates[0])
        plt.plot(
            initial_position[0],
            initial_position[1],
            "o",
            color="grey",
            markersize=marker_size,
        )
        final_position = (x_coordinates[-1], y_coordinates[-1])
        plt.plot(
            final_position[0],
            final_position[1],
            "o",
            color="red",
            markersize=marker_size,
        )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    legend_entries = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="grey",
            markeredgecolor="grey",
            markersize=marker_size,
            linestyle="None",
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=marker_size,
            linestyle="None",
            label="End",
        ),
    ]
    plt.legend(handles=legend_entries, loc="best")
    filename = "trajectories.pdf"
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def plot_coordinates(coordinates: Iterable) -> None:
    """coordinates: array of points in R^2. Example: [[0,1], [0.5,1], [1,1]]"""
    for point in coordinates:
        plt.plot(point[0], point[1], "o", markersize=5, color="grey")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Initial conditions")
    plt.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def set_custom_xtick(y_vals: np.ndarray, at_index: int) -> None:
    ax = plt.gca()
    default_ticks = ax.get_xticks()
    default_ticks = default_ticks[(default_ticks >= 0) & (default_ticks <= y_vals.size)]
    all_ticks = np.unique(np.append(default_ticks, at_index))
    ax.set_xticks(all_ticks)
    for tick_value, tick_label in zip(
        ax.get_xticks(), ax.get_xticklabels(), strict=True
    ):
        if np.isclose(tick_value, at_index):
            tick_label.set_color("tab:red")


def plot_eigenvalues_vs_index(
    eigenvalues: np.ndarray, subfolder: Optional[str] = None
) -> None:
    """
    Plots the eigenvalues in ascending order with respect to their natural index.
    Also highlights the largest eigengap to identify its index.
    """
    vals = np.sort(np.asarray(eigenvalues).ravel())
    if vals.size < 2:
        raise ValueError(
            "At least 2 eigenvalues are required to calculate the eigengap."
        )

    indices = np.arange(vals.size)
    eigengaps = np.diff(vals)
    largest_eigengap_index = int(np.argmax(eigengaps))
    k = largest_eigengap_index

    plt.figure(figsize=(9, 5))
    plt.plot(
        indices,
        vals,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Eigenvalues",
    )
    plt.axvline(
        k,
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=r"Largest eigengap is at index $k=$" + f"{k}",
    )
    plt.axvline(k + 1, color="tab:red", linestyle="--", alpha=0.5)
    plt.plot([k, k + 1], [vals[k], vals[k + 1]], color="tab:red", linewidth=2.5)
    set_custom_xtick(vals, at_index=k)
    plt.annotate(
        "Largest eigengap",
        xy=(k + 0.5, 0.5 * (vals[k] + vals[k + 1])),
        xytext=(k + 4, 0.4 * (vals[k] + vals[k + 1])),
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        fontsize=12,
        color="tab:red",
    )
    plt.xlabel(r"Index $k$")
    plt.ylabel(r"Eigenvalue ($\lambda_{k}$)")
    plt.title("Eigenvalues sorted by index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    filename = "eigenvalues_vs_index.pdf"
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def plot_eigengaps_vs_index(
    eigenvalues: np.ndarray, subfolder: Optional[str] = None
) -> None:
    """
    Plots the eigengaps (consecutive differences of eigenvalues) with
    respect to their natural index and highlights the largest eigengap.
    """
    vals = np.sort(np.asarray(eigenvalues).ravel())
    if vals.size < 2:
        raise ValueError(
            "At least 2 eigenvalues are required to calculate the eigengap."
        )

    eigengaps = np.diff(vals)
    indices = np.arange(eigengaps.size)
    largest_eigengap_index = int(np.argmax(eigengaps))
    largest_eigengap_value = float(eigengaps[largest_eigengap_index])
    k = largest_eigengap_index

    plt.figure(figsize=(9, 5))
    plt.plot(
        indices,
        eigengaps,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Eigengaps",
    )
    plt.axvline(
        k,
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=r"Largest eigengap is at index $k=$" + f"{k}",
    )
    plt.scatter([k], [largest_eigengap_value], color="tab:red", zorder=3)
    set_custom_xtick(eigengaps, at_index=k)
    plt.annotate(
        "Largest eigengap",
        xy=(k, largest_eigengap_value),
        xytext=(k + 3, 0.95 * largest_eigengap_value),
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        fontsize=12,
        color="tab:red",
    )
    plt.xlabel(r"Index $k$")
    plt.ylabel(r"Eigengap ($\lambda_{k+1} - \lambda_{k}$)")
    plt.title("Eigengaps with respect to its index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    filename = "eigengaps_vs_index.pdf"
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def plot_clusters(
    initial_conditions: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    sparsification_radius: float,
    sparsification_percent: float,
    params: SpectralClusteringConfig,
    subfolder: Optional[str] = None,
    filename_prefix: str = "",
) -> None:
    """
    Classifies trajectories into clusters. I.e., plots the initial
    condition for each trajectory and colors them by their labeled cluster.
    """
    num_trajectories = len(initial_conditions)
    for cluster_id in range(num_clusters):
        indices = np.where(labels == cluster_id)
        if len(indices[0]) > 0:
            plt.scatter(
                initial_conditions[indices, 0],
                initial_conditions[indices, 1],
                s=30,
                label=cluster_id,
            )
    plt.title("Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.gca().set_aspect("equal")
    clustering_metrics = (
        f"{num_clusters} clusters, {num_trajectories} trajectories, "
        f"{params.t_steps} time steps,"
        "\n"
        f"end time = {params.t_span[-1]:.1f}s, "
        f"sparsification = {sparsification_percent * 100:.0f}%"
        f", sparsification radius = {sparsification_radius:.2f}"
    )
    plt.text(
        0.5,
        -0.18,
        clustering_metrics,
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    filename = (
        filename_prefix + f"clusters={num_clusters}"
        f"_traj={num_trajectories}"
        f"_tsteps={params.t_steps}"
        f"_t_end={params.t_span[-1]:.1f}"
        f"_tol={sparsification_radius:.1f}"
        f"_sparse={sparsification_percent * 100:.0f}"
        ".pdf"
    )
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def make_patch_spines_invisible(ax) -> None:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def highlight_local_maxima(
    eigengap_local_maxima_indices: Optional[list[int]],
    result: SpectralAnalysisResult,
    host: Axes,
    lines: list[Line2D],
    labels: list[str],
) -> None:
    """
    Helper function that plots medium-sized orange circles on the eigengap vs
    sparsity plot to highlight local maxima of eigengaps, i.e.,
    the relative maxima of the differences between consecutive eigenvalues.
    """
    if not eigengap_local_maxima_indices:
        return
    for index in eigengap_local_maxima_indices:
        host.plot(
            result.sparsification_radii[index],
            result.normalized_eigengaps[index],
            marker="o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="tab:orange",
            markeredgewidth=2.1,
        )
    local_maxima_marker = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=12,
        markerfacecolor="none",
        markeredgecolor="tab:orange",
        markeredgewidth=2.1,
    )
    lines.append(local_maxima_marker)
    labels.append("Local maxima")


def plot_eigengaps_vs_sparsity(
    result: SpectralAnalysisResult,
    params: SpectralClusteringConfig,
    eigengap_local_maxima_indices: Optional[list[int]] = None,
    subfolder: Optional[str] = None,
) -> None:
    """
    Plots the eigengap, number of clusters, and sparsification percentage
    as a function of the sparsification radius. Highlights the relative
    maxima (local peaks) of the differences between consecutive eigenvalues.
    """
    fig, host = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.1))

    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)
    color_gap = "tab:blue"
    color_clust = "tab:red"
    color_sparse = "tab:green"
    (p1,) = host.plot(
        result.sparsification_radii,
        result.normalized_eigengaps,
        marker=".",
        color=color_gap,
        label="Normalized eigengap",
    )
    (p2,) = par1.plot(
        result.sparsification_radii,
        result.nums_clusters,
        marker=".",
        color=color_clust,
        label="Number of clusters",
    )
    (p3,) = par2.plot(
        result.sparsification_radii,
        result.sparsification_percents,
        marker=".",
        color=color_sparse,
        label="Sparsification (%)",
    )
    par2.set_ylim(0, 1.0)
    par2.yaxis.set_major_formatter(PercentFormatter(1.0))
    host.set_xlabel("Sparsification radius")
    host.set_ylabel("Normalized eigengap", color=color_gap)
    par1.set_ylabel("Number of clusters", color=color_clust)
    par2.set_ylabel("Sparsification (%)", color=color_sparse)
    host.tick_params(axis="y", labelcolor=color_gap)
    par1.tick_params(axis="y", labelcolor=color_clust)
    par2.tick_params(axis="y", labelcolor=color_sparse)
    host.yaxis.get_offset_text().set_horizontalalignment("left")
    host.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    colors_stats = iter(plt.rcParams["axes.prop_cycle"])
    next(colors_stats)
    for metric_name, metric_value in result.weight_statistics.items():
        if metric_name != "pes_max":
            host.axvline(
                x=metric_value,
                linestyle="--",
                alpha=0.6,
                color=next(colors_stats)["color"],
                label=f"{metric_name} = {metric_value:.2f}",
            )

    host.set_title(r"Eigengap and cluster count vs sparsification radius")
    host.grid(True, alpha=0.3)
    lines = [p1, p2, p3]
    labels: list[str] = [str(line.get_label()) for line in lines]
    for line in host.get_lines():
        if line not in lines:
            lines.append(line)
            labels.append(str(line.get_label()))

    highlight_local_maxima(eigengap_local_maxima_indices, result, host, lines, labels)

    host.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    filename = (
        "eigengap_vs_radii"
        f"-max_clusters={params.max_clusters}"
        f"_num_radii={len(result.sparsification_radii)}"
        f"_t_end={params.t_span[1]:.1f}"
        ".pdf"
    )
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()
