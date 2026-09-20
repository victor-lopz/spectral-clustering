from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SpectralClusteringConfig:
    num_dimensions: int = 2
    t_span: Tuple[float, float] = (0, 4 * np.pi)
    t_steps: int = 300
    x_min: float = -1.6
    x_max: float = 1.6
    y_min: float = -1.0
    y_max: float = 1.0
    grid_spacing: float = 0.04
    max_clusters: int = 50
    num_radii: int = 50
    t_values: np.ndarray = field(init=False)

    def __post_init__(self):
        self.t_values = np.linspace(self.t_span[0], self.t_span[1], self.t_steps)


@dataclass
class SpectralAnalysisResult:
    sparsification_radii: np.ndarray
    eigengaps: List[float] = field(default_factory=list)
    normalized_eigengaps: List[float] = field(default_factory=list)
    nums_clusters: List[int] = field(default_factory=list)
    weight_statistics: Dict[str, float] = field(default_factory=dict)
    sparsification_percents: List[float] = field(default_factory=list)
    veps_list: List[np.ndarray] = field(default_factory=list)
