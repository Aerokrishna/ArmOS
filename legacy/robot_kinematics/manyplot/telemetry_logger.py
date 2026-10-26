from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


class TelemetryLogger:
    def __init__(self) -> None:
        self._joint_rows: list[dict[str, Any]] = []
        self._task_rows: list[dict[str, Any]] = []

    @staticmethod
    def _as_list(values: Any) -> list[float]:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return [float(array)]
        return array.reshape(-1).tolist()

    @staticmethod
    def _write_csv(file_path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, list) else value
                        for key, value in row.items()
                    }
                )

    def collect_joint_data(self, joint_pose: Any, joint_torque: Any, timestamp: float) -> None:
        self._joint_rows.append(
            {
                "timestamp": float(timestamp),
                "pos": self._as_list(joint_pose),
                "torque": self._as_list(joint_torque),
            }
        )

    def collect_task_space_data(self, task_pose: Any, task_wrench: Any, timestamp: float) -> None:
        self._task_rows.append(
            {
                "timestamp": float(timestamp),
                "pose": self._as_list(task_pose),
                "wrench": self._as_list(task_wrench),
            }
        )

    def save_csv(self, file_path: str | Path) -> None:
        self._write_csv(file_path, self._joint_rows, ["timestamp", "pos", "torque"])

    def save_task_csv(self, file_path: str | Path) -> None:
        self._write_csv(file_path, self._task_rows, ["timestamp", "pose", "wrench"])

    def clear(self) -> None:
        self._joint_rows.clear()
        self._task_rows.clear()