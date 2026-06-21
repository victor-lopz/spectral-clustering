import os
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from src.datatypes import ParametresGenerals, SpectralAnalysisResult


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


def grafica_trajectories(
    trajectories: np.ndarray, subfolder: str | None = None, titol: str | None = None
) -> None:
    if titol:
        plt.title(titol)
    for trajectoria in trajectories:
        coordenades_x = trajectoria[:, 0]
        coordenades_y = trajectoria[:, 1]
        plt.plot(coordenades_x, coordenades_y)
    mida_punt = 5
    for trajectoria in trajectories:
        coordenades_x = trajectoria[:, 0]
        coordenades_y = trajectoria[:, 1]
        pos_inicial = (coordenades_x[0], coordenades_y[0])
        plt.plot(
            pos_inicial[0], pos_inicial[1], "o", color="grey", markersize=mida_punt
        )
        pos_final = (coordenades_x[-1], coordenades_y[-1])
        plt.plot(pos_final[0], pos_final[1], "o", color="red", markersize=mida_punt)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.gca().set_aspect("equal", adjustable="box")
    punts_llegenda = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="grey",
            markeredgecolor="grey",
            markersize=mida_punt,
            linestyle="None",
            label="Inici",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=mida_punt,
            linestyle="None",
            label="Final",
        ),
    ]
    plt.legend(handles=punts_llegenda, loc="best")
    filename = "trajectories.pdf"
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def grafica_punts(punts: Iterable) -> None:
    """punts: conjunt de punts a R^2. Exemple: [[0,1], [0.5,1], [1,1]]"""
    for punt in punts:
        plt.plot(punt[0], punt[1], "o", markersize=5, color="grey")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Condicions inicials")
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


def grafica_eigenvalues_vs_index(
    eigenvalues: np.ndarray, subfolder: str | None = None
) -> None:
    """
    Grafica els valors propis ordenats de menor a major respecte el seu index natural.
    També destaca el major eigengap per identificar visualment k.
    """
    vals = np.sort(np.asarray(eigenvalues).ravel())
    if vals.size < 2:
        raise ValueError("Calen com a minim 2 valors propis per calcular l'eigengap.")

    indexes = np.arange(vals.size)
    gaps = np.diff(vals)
    max_gap_pos = int(np.argmax(gaps))
    k = max_gap_pos

    plt.figure(figsize=(9, 5))
    plt.plot(
        indexes,
        vals,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Valors propis",
    )
    plt.axvline(
        k,
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=r"Salt màxim en $k=$" + f"{k}",
    )
    plt.axvline(k + 1, color="tab:red", linestyle="--", alpha=0.5)
    plt.plot([k, k + 1], [vals[k], vals[k + 1]], color="tab:red", linewidth=2.5)
    set_custom_xtick(vals, at_index=k)
    plt.annotate(
        "Salt màxim",
        xy=(k + 0.5, 0.5 * (vals[k] + vals[k + 1])),
        xytext=(k + 4, 0.4 * (vals[k] + vals[k + 1])),
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        fontsize=12,
        color="tab:red",
    )
    plt.xlabel(r"Índex $k$")
    plt.ylabel(r"Valor propi ($\lambda_{k}$)")
    plt.title("Valors propis respecte al seu índex")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    filename = "eigenvalues_vs_index.pdf"
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def grafica_eigengaps_vs_index(
    eigenvalues: np.ndarray, subfolder: str | None = None
) -> None:
    """
    Grafica els eigengaps (diferències consecutives de valors propis ordenats)
    respecte al seu index natural i destaca el màxim eigengap.
    """
    vals = np.sort(np.asarray(eigenvalues).ravel())
    if vals.size < 2:
        raise ValueError(
            "Calen com a minim 2 valors propis per calcular els eigengaps."
        )

    gaps = np.diff(vals)
    indexes = np.arange(gaps.size)
    max_gap_pos = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_pos])
    k = max_gap_pos

    plt.figure(figsize=(9, 5))
    plt.plot(
        indexes,
        gaps,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Salts espectrals",
    )
    plt.axvline(
        k,
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=r"Salt màxim en $k=$" + f"{k}",
    )
    plt.scatter([k], [max_gap], color="tab:red", zorder=3)
    set_custom_xtick(gaps, at_index=k)
    plt.annotate(
        "Salt màxim",
        xy=(k, max_gap),
        xytext=(k + 3, 0.95 * max_gap),
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        fontsize=12,
        color="tab:red",
    )
    plt.xlabel(r"Índex $k$")
    plt.ylabel(r"Salt espectral ($\lambda_{k+1} - \lambda_{k}$)")
    plt.title("Salts espectrals respecte l'índex")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    filename = "eigengaps_vs_index.pdf"
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def grafica_clusters(
    condicions_inicials: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    radi_esparsificacio: float,
    percent_esparsificacio: float,
    params: ParametresGenerals,
    subfolder: str | None = None,
    filename_prefix: str = "",
) -> None:

    num_trajectories = len(condicions_inicials)
    for cluster_id in range(num_clusters):
        indices = np.where(labels == cluster_id)
        if len(indices[0]) > 0:
            plt.scatter(
                condicions_inicials[indices, 0],
                condicions_inicials[indices, 1],
                s=30,
                label=cluster_id,
            )
    plt.title("Clústers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.gca().set_aspect("equal")
    descripcio = (
        f"{num_clusters} clústers, {num_trajectories} trajectòries, "
        f"{params.t_steps} passes de temps,"
        "\n"
        f"temps final = {params.t_span[-1]:.1f}s, "
        f"esparsificació = {percent_esparsificacio * 100:.0f}%"
        f", radi = {radi_esparsificacio:.2f}"
    )
    plt.text(
        0.5,
        -0.18,
        descripcio,
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    # plt.figtext(0.5, 0.01, descripcio, ha='center', fontsize=11)
    # plt.subplots_adjust(bottom=0.1)
    filename = (
        filename_prefix + f"clusters={num_clusters}"
        f"_traj={num_trajectories}"
        f"_tsteps={params.t_steps}"
        f"_t_end={params.t_span[-1]:.1f}"
        f"_tol={radi_esparsificacio:.1f}"
        f"_sparse={percent_esparsificacio * 100:.0f}"
        ".pdf"
    )
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()


def grafica_eigengaps_vs_radi(
    result: SpectralAnalysisResult,
    params: ParametresGenerals,
    indexs_max_rel: list[int] | None = None,
    subfolder: str | None = None,
) -> None:
    """
    Grafica el nombre de clusters, l'eigengap i el percentatge d'esparsificació
    en funció del radi d'esparsificació. Encercla els màxims relatius de les diferències entre VAPs consecutius.
    """
    fig, host = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.1))

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)
    color_gap = "tab:blue"
    color_clust = "tab:red"
    color_sparse = "tab:green"
    (p1,) = host.plot(
        result.radis,
        result.normalized_eigengaps,
        marker=".",
        color=color_gap,
        label="Eigengap normalitzat",
    )
    (p2,) = par1.plot(
        result.radis,
        result.nums_clusters,
        marker=".",
        color=color_clust,
        label="Nombre de clústers",
    )
    (p3,) = par2.plot(
        result.radis,
        result.sparsificacions,
        marker=".",
        color=color_sparse,
        label="Esparsificació (%)",
    )
    par2.set_ylim(0, 1.0)
    par2.yaxis.set_major_formatter(PercentFormatter(1.0))
    host.set_xlabel("Radi d'esparsificació")
    host.set_ylabel(
        "Diferència màxima normalitzada entre VAPs consecutius", color=color_gap
    )
    par1.set_ylabel("Nombre de clústers", color=color_clust)
    par2.set_ylabel("Esparsificació (%)", color=color_sparse)
    host.tick_params(axis="y", labelcolor=color_gap)
    par1.tick_params(axis="y", labelcolor=color_clust)
    par2.tick_params(axis="y", labelcolor=color_sparse)
    host.yaxis.get_offset_text().set_horizontalalignment("left")
    host.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    colors_stats = iter(plt.rcParams["axes.prop_cycle"])
    next(colors_stats)
    for nom, valor in result.estadistics.items():
        if nom != "pes_max":
            host.axvline(
                x=valor,
                linestyle="--",
                alpha=0.6,
                color=next(colors_stats)["color"],
                label=f"{nom} = {valor:.2f}",
            )

    host.set_title(r"Eigengap i nombre de clústers vs radi d'esparsificació")
    host.grid(True, alpha=0.3)
    lines = [p1, p2, p3]
    labels: list[str] = [str(l.get_label()) for l in lines]
    for l in host.get_lines():
        if l not in lines:
            lines.append(l)
            labels.append(str(l.get_label()))

    def destaca_max_rels():
        if indexs_max_rel is None:
            return
        for index in indexs_max_rel:
            host.plot(
                result.radis[index],
                result.normalized_eigengaps[index],
                marker="o",
                markersize=12,
                markerfacecolor="none",
                markeredgecolor="tab:orange",
                markeredgewidth=2.1,
            )
        if len(indexs_max_rel) > 0:
            proxy_circle = Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=12,
                markerfacecolor="none",
                markeredgecolor="tab:orange",
                markeredgewidth=2.1,
            )
            lines.append(proxy_circle)
            labels.append("Màxim relatiu")

    destaca_max_rels()
    host.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    filename = (
        f"eigengap_vs_radi-max_clusters={params.max_clusters}_radis={len(result.radis)}"
        f"_t_end={params.t_span[1]:.1f}.pdf"
    )
    plt.savefig(get_output_path(filename, subfolder), bbox_inches="tight")
    plt.show()
