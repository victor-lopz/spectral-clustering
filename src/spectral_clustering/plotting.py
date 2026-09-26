from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from .datatypes import SpectralAnalysisResult, SpectralClusteringConfig

ALLOWED_EXTENSIONS = frozenset((".pdf", ".png", ".jpg", ".jpeg", ".svg"))


def is_valid(filename: str) -> bool:
    """Checks if a filename has a proper base name and an allowed extension."""
    try:
        file_path = Path(filename)
    except TypeError:
        return False
    if file_path.parent != Path("."):
        return False
    extension = file_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False
    return file_path.stem


def save_figure_optionally(
    filename: str | None = None,
    default_filename: str | None = None,
    output_dir: str | Path | None = None,
    subfolder: str | None = None,
) -> None:
    """
    Saves the current plt figure if output_dir and a (default)filename are provided.
    Otherwise, does nothing.

    Args:
        filename: name of the file to save with extension.
        output_dir: root directory for outputs.
        subfolder: optional subdirectory name.
    """
    if not output_dir:
        return
    if not is_valid(filename):
        if not is_valid(default_filename):
            return
        filename = default_filename
    base_path = Path(output_dir)
    if subfolder:
        base_path = base_path / subfolder
    base_path.mkdir(parents=True, exist_ok=True)
    file_path = base_path / filename
    plt.savefig(file_path, bbox_inches="tight")


def plot_trajectory_paths(
    trajectories: np.ndarray,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    subfolder: str | None = None,
    plot_title: str | None = None,
) -> Axes:
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
    default_filename = "trajectories.pdf"
    save_figure_optionally(filename, default_filename, output_dir, subfolder)
    return plt.gca()


def plot_coordinates(coordinates: Iterable) -> Axes:
    """coordinates: array of points in R^2. Example: [[0,1], [0.5,1], [1,1]]"""
    for point in coordinates:
        plt.plot(point[0], point[1], "o", markersize=5, color="grey")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Initial conditions")
    plt.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    return plt.gca()


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
    eigenvalues: np.ndarray,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    subfolder: str | None = None,
    plot_title: str | None = None,
) -> Axes:
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
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        fontsize=12,
        color="tab:red",
    )
    plt.xlabel(r"Index $k$")
    plt.ylabel(r"Eigenvalue ($\lambda_{k}$)")
    if plot_title is None:
        plot_title = "Eigenvalues with respect to their index"
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    default_filename = "eigenvalues_vs_index.pdf"
    save_figure_optionally(filename, default_filename, output_dir, subfolder)
    return plt.gca()


def plot_eigengaps_vs_index(
    eigenvalues: np.ndarray,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    subfolder: str | None = None,
    plot_title: str | None = None,
) -> Axes:
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
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        fontsize=12,
        color="tab:red",
    )
    plt.xlabel(r"Index $k$")
    plt.ylabel(r"Eigengap ($\lambda_{k+1} - \lambda_{k}$)")
    if plot_title is None:
        plot_title = "Eigengaps with respect to its index"
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    default_filename = "eigengaps_vs_index.pdf"
    save_figure_optionally(filename, default_filename, output_dir, subfolder)
    return plt.gca()


def plot_clusters(
    initial_conditions: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    sparsification_radius: float,
    sparsification_percent: float,
    params: SpectralClusteringConfig,
    output_dir: str | Path | None = None,
    subfolder: str | None = None,
    filename: str | None = None,
    plot_title: str | None = None,
) -> Axes:
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
    if plot_title is None:
        plot_title = "Clusters"
    plt.title(plot_title)
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
    default_filename = (
        f"clusters={num_clusters}"
        f"_sparse={sparsification_percent * 100:.0f}"
        f"_tol={sparsification_radius:.1f}"
        f"_traj={num_trajectories}"
        f"_tsteps={params.t_steps}"
        f"_t_end={params.t_span[-1]:.1f}.pdf"
    )
    save_figure_optionally(filename, default_filename, output_dir, subfolder)
    return plt.gca()


def make_patch_spines_invisible(ax) -> None:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def highlight_local_maxima(
    eigengap_local_maxima_indices: list[int] | None,
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
    eigengap_local_maxima_indices: list[int] | None = None,
    output_dir: str | Path | None = None,
    subfolder: str | None = None,
    filename: str | None = None,
    plot_title: str | None = None,
) -> Axes:
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
        if metric_name != "max_weight":
            host.axvline(
                x=metric_value,
                linestyle="--",
                alpha=0.6,
                color=next(colors_stats)["color"],
                label=f"{metric_name} = {metric_value:.2f}",
            )
    if plot_title is None:
        plot_title = "Eigengap and cluster count vs sparsification radius"
    host.set_title(plot_title)
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
    default_filename = (
        "eigengap_vs_radii"
        f"-max_clusters={params.max_clusters}"
        f"_num_radii={len(result.sparsification_radii)}"
        f"_t_end={params.t_span[1]:.1f}.pdf"
    )
    save_figure_optionally(filename, default_filename, output_dir, subfolder)
    return plt.gca()
