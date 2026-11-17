from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from .basis_factory import CoherenceBasisFactory
from .coherence_extractor import RelativeEntropyCoherenceExtractor
from .config import BasisConfig, DatasetConfig, PathConfig
from .ppt_labeler import PPTLabeler
from .state_factory import BipartiteStateFactory

if TYPE_CHECKING:  # pragma: no cover
    from qutip import Qobj

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RawStateRecord:
    family_name: str
    index: int
    label: int
    state: "Qobj"


@dataclass(slots=True)
class DatasetArtifact:
    features: np.ndarray
    labels: np.ndarray
    family_ids: np.ndarray
    family_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    manifest: dict[str, Any]
    raw_state_records: dict[str, list[RawStateRecord]] | None = None


class DatasetBuilder:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        basis_config: BasisConfig,
        path_config: PathConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.dataset_config = dataset_config
        self.basis_config = basis_config
        self.path_config = path_config
        self.logger = logger or LOGGER
        self.labeler = PPTLabeler()
        self.bases = CoherenceBasisFactory(basis_config).build()
        self.extractor = RelativeEntropyCoherenceExtractor(self.bases)
        self.state_factory = BipartiteStateFactory(dataset_config, labeler=self.labeler)

    def build(self) -> DatasetArtifact:
        feature_rows: list[np.ndarray] = []
        labels: list[int] = []
        family_ids: list[int] = []
        family_names: list[str] = []
        family_name_to_id: dict[str, int] = {}
        class_counts: Counter[int] = Counter()
        family_counts: Counter[str] = Counter()
        raw_state_records: dict[str, list[RawStateRecord]] | None = None

        if self.path_config.save_raw_states:
            raw_state_records = defaultdict(list)

        log_interval = max(1000, self.dataset_config.total_samples // 100)

        for index, (family_name, state) in enumerate(self.state_factory.iter_all_states(), start=1):
            if family_name not in family_name_to_id:
                family_name_to_id[family_name] = len(family_names)
                family_names.append(family_name)

            label = self.labeler.label(state)
            features = self.extractor.extract_features(state)

            feature_rows.append(features)
            labels.append(label)
            family_ids.append(family_name_to_id[family_name])
            class_counts[label] += 1
            family_counts[family_name] += 1

            if raw_state_records is not None and self._should_capture_raw_state(family_name, raw_state_records):
                raw_state_records[family_name].append(
                    RawStateRecord(
                        family_name=family_name,
                        index=family_counts[family_name] - 1,
                        label=label,
                        state=state,
                    )
                )

            if index % log_interval == 0 or index == self.dataset_config.total_samples:
                self.logger.info("Generated and featurized %s/%s states.", index, self.dataset_config.total_samples)

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_config": self.dataset_config.to_dict(),
            "basis_config": self.basis_config.to_dict(),
            "path_config": self.path_config.to_dict(),
            "family_counts": dict(family_counts),
            "class_counts": {"separable": class_counts.get(0, 0), "entangled": class_counts.get(1, 0)},
            "feature_names": list(self.extractor.feature_names),
            "total_samples": len(feature_rows),
        }

        return DatasetArtifact(
            features=np.asarray(feature_rows, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int64),
            family_ids=np.asarray(family_ids, dtype=np.int64),
            family_names=tuple(family_names),
            feature_names=self.extractor.feature_names,
            manifest=manifest,
            raw_state_records=dict(raw_state_records) if raw_state_records is not None else None,
        )

    def _should_capture_raw_state(
        self,
        family_name: str,
        raw_state_records: dict[str, list[RawStateRecord]],
    ) -> bool:
        limit = self.path_config.raw_state_limit_per_family
        if limit == 0:
            return True
        return len(raw_state_records[family_name]) < limit
