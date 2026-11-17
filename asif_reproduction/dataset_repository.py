from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._dependencies import require_qutip
from .config import PathConfig
from .dataset_builder import DatasetArtifact
from .provenance import dataset_logical_fingerprint, sha256_file


class DatasetRepository:
    def __init__(self, path_config: PathConfig) -> None:
        self.path_config = path_config

    def save(self, dataset: DatasetArtifact) -> None:
        self.path_config.ensure_directories()
        np.savez_compressed(
            self.path_config.dataset_npz_path,
            X=dataset.features,
            y=dataset.labels,
            family_ids=dataset.family_ids,
            family_names=np.asarray(dataset.family_names, dtype=str),
            feature_names=np.asarray(dataset.feature_names, dtype=str),
        )
        dataset.manifest["dataset_artifact"] = {
            "path": str(self.path_config.dataset_npz_path),
            "sha256": sha256_file(self.path_config.dataset_npz_path),
            "logical_fingerprint_sha256": dataset_logical_fingerprint(self.path_config.dataset_npz_path),
        }
        with self.path_config.dataset_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(dataset.manifest, handle, indent=2, sort_keys=True)

        if self.path_config.save_raw_states and dataset.raw_state_records:
            self._save_raw_states(dataset.raw_state_records)

    def load(self) -> DatasetArtifact:
        if not self.path_config.dataset_npz_path.exists():
            raise FileNotFoundError(f"Dataset archive not found: {self.path_config.dataset_npz_path}")
        if not self.path_config.dataset_manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {self.path_config.dataset_manifest_path}")

        data = np.load(self.path_config.dataset_npz_path)
        with self.path_config.dataset_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        return DatasetArtifact(
            features=data["X"],
            labels=data["y"],
            family_ids=data["family_ids"],
            family_names=tuple(data["family_names"].tolist()),
            feature_names=tuple(data["feature_names"].tolist()),
            manifest=manifest,
            raw_state_records=None,
        )

    def load_manifest(self) -> dict[str, Any]:
        with self.path_config.dataset_manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_raw_states(self, raw_state_records: dict[str, list[Any]]) -> None:
        qutip = require_qutip()
        for family_name, records in raw_state_records.items():
            family_dir = self.path_config.raw_state_dir / family_name
            family_dir.mkdir(parents=True, exist_ok=True)
            for record in records:
                stem = family_dir / f"{family_name}_{record.index:06d}_label_{record.label}"
                qutip.qsave(record.state, str(stem))
