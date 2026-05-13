
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class ParametresGenerals:
    dimensio: int = 2
    t_span: Tuple[float, float] = (0, 4 * np.pi)
    t_steps: int = 300
    x_min: float = -1.6
    x_max: float = 1.6
    y_min: float = -1.0
    y_max: float = 1.0
    espai_entre_punts: float = 0.05
    max_clusters: int = 50
    num_radis: int = 50
    t_valors: np.ndarray = field(init=False)

    def __post_init__(self):
        self.t_valors = np.linspace(self.t_span[0], self.t_span[1], self.t_steps)

@dataclass
class SpectralAnalysisResult:
    radis: np.ndarray
    eigengaps: List[float] = field(default_factory=list)
    normalized_eigengaps: List[float] = field(default_factory=list)
    nums_clusters: List[int] = field(default_factory=list)
    estadistics: Dict[str, float] = field(default_factory=dict)
    sparsificacions: List[float] = field(default_factory=list)
    veps_list: List[np.ndarray] = field(default_factory=list)
