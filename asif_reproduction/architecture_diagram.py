from __future__ import annotations

import os
from pathlib import Path

from ._dependencies import require_matplotlib


class SystemArchitectureDiagram:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def render(self) -> None:
        pyplot, patches = self._plot_modules()

        figure, axis = pyplot.subplots(figsize=(14, 8))
        axis.set_xlim(0, 14)
        axis.set_ylim(0, 10)
        axis.axis("off")

        self._box(axis, patches, 0.8, 7.8, 2.8, 1.2, "config.py\nEditable run parameters", "#d8f3dc")
        self._box(axis, patches, 0.8, 5.8, 2.8, 1.2, "main.py\nTop-level executor", "#b7e4c7")

        self._box(axis, patches, 4.2, 7.8, 3.0, 1.2, "ExperimentRunner", "#cfe8ff")
        self._box(axis, patches, 4.2, 5.8, 3.0, 1.2, "DatasetBuilder", "#cfe8ff")
        self._box(axis, patches, 4.2, 3.8, 3.0, 1.2, "ANNExperiment", "#cfe8ff")

        self._box(axis, patches, 8.0, 8.3, 2.6, 1.0, "CoherenceBasisFactory", "#ffe5ec")
        self._box(axis, patches, 8.0, 7.0, 2.6, 1.0, "BipartiteStateFactory", "#ffe5ec")
        self._box(axis, patches, 8.0, 5.7, 2.6, 1.0, "PPTLabeler", "#ffe5ec")
        self._box(axis, patches, 8.0, 4.4, 2.6, 1.0, "RelativeEntropy\nCoherenceExtractor", "#ffe5ec")
        self._box(axis, patches, 8.0, 3.1, 2.6, 1.0, "DatasetRepository", "#ffe5ec")
        self._box(axis, patches, 8.0, 1.8, 2.6, 1.0, "ExperimentPlotter\n+ ThesisReportWriter", "#ffe5ec")

        self._box(axis, patches, 11.4, 7.6, 2.0, 1.0, "datasets/", "#fff3bf")
        self._box(axis, patches, 11.4, 6.2, 2.0, 1.0, "models/", "#fff3bf")
        self._box(axis, patches, 11.4, 4.8, 2.0, 1.0, "metrics/", "#fff3bf")
        self._box(axis, patches, 11.4, 3.4, 2.0, 1.0, "plots/", "#fff3bf")
        self._box(axis, patches, 11.4, 2.0, 2.0, 1.0, "THESIS_RESULTS.md", "#fff3bf")

        self._arrow(axis, 3.6, 8.4, 4.2, 8.4)
        self._arrow(axis, 2.2, 7.8, 2.2, 7.0)
        self._arrow(axis, 3.6, 6.4, 4.2, 6.4)
        self._arrow(axis, 5.7, 7.8, 5.7, 7.0)
        self._arrow(axis, 5.7, 5.8, 5.7, 5.0)

        self._arrow(axis, 7.2, 8.4, 8.0, 8.8)
        self._arrow(axis, 7.2, 6.4, 8.0, 7.5)
        self._arrow(axis, 7.2, 6.4, 8.0, 6.2)
        self._arrow(axis, 7.2, 6.4, 8.0, 4.9)
        self._arrow(axis, 7.2, 4.4, 8.0, 3.6)
        self._arrow(axis, 7.2, 4.4, 8.0, 2.3)

        self._arrow(axis, 10.6, 3.6, 11.4, 8.1)
        self._arrow(axis, 10.6, 3.6, 11.4, 6.7)
        self._arrow(axis, 10.6, 2.3, 11.4, 5.3)
        self._arrow(axis, 10.6, 2.3, 11.4, 3.9)
        self._arrow(axis, 10.6, 2.3, 11.4, 2.5)

        axis.set_title("System Architecture for the Entanglement-Reproduction Pipeline", fontsize=16, pad=20)
        figure.tight_layout()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.output_dir / "system_architecture.png", dpi=180)
        figure.savefig(self.output_dir / "system_architecture.pdf")
        pyplot.close(figure)

    def _plot_modules(self):
        cache_dir = self.output_dir / ".matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        matplotlib = require_matplotlib()
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        from matplotlib import patches

        return pyplot, patches

    @staticmethod
    def _box(axis, patches, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
        rectangle = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#355070",
            facecolor=color,
        )
        axis.add_patch(rectangle)
        axis.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center", fontsize=10)

    @staticmethod
    def _arrow(axis, x_start: float, y_start: float, x_end: float, y_end: float) -> None:
        axis.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(x_start, y_start),
            arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#355070"},
        )
