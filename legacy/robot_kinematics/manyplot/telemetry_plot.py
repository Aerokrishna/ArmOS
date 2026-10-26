from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _parse_vector(cell: str) -> np.ndarray:
	if cell is None:
		return np.array([], dtype=float)

	text = str(cell).strip()
	if not text:
		return np.array([], dtype=float)

	try:
		value = json.loads(text)
	except json.JSONDecodeError:
		value = ast.literal_eval(text)

	return np.asarray(value, dtype=float)


def _load_telemetry_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
	with csv_path.open(newline="") as handle:
		reader = csv.DictReader(handle)
		rows = list(reader)

	if not rows:
		raise ValueError(f"No telemetry rows found in {csv_path}")

	fieldnames = reader.fieldnames or []
	if {"pos", "torque"}.issubset(fieldnames):
		position_key = "pos"
		force_key = "torque"
		data_label = "joint"
	elif {"pose", "wrench"}.issubset(fieldnames):
		position_key = "pose"
		force_key = "wrench"
		data_label = "task"
	else:
		raise ValueError(
			f"Unsupported CSV columns in {csv_path}. Expected pos/torque or pose/wrench."
		)

	timestamp = np.asarray([float(row["timestamp"]) for row in rows], dtype=float)
	position = np.vstack([_parse_vector(row[position_key]) for row in rows])
	force = np.vstack([_parse_vector(row[force_key]) for row in rows])

	if position.ndim == 1:
		position = position[:, None]
	if force.ndim == 1:
		force = force[:, None]

	order = np.argsort(timestamp)
	timestamp = timestamp[order]
	position = position[order]
	force = force[order]

	unique_timestamp, unique_index = np.unique(timestamp, return_index=True)
	timestamp = unique_timestamp
	position = position[unique_index]
	force = force[unique_index]

	if timestamp.size < 2:
		raise ValueError("Need at least two samples to compute derivatives")

	return timestamp, position, force, data_label


def _differentiate(values: np.ndarray, timestamp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	edge_order = 2 if timestamp.size > 2 else 1
	velocity = np.gradient(values, timestamp, axis=0, edge_order=edge_order)
	acceleration = np.gradient(velocity, timestamp, axis=0, edge_order=edge_order)
	jerk = np.gradient(acceleration, timestamp, axis=0, edge_order=edge_order)
	return velocity, acceleration, jerk


def _plot_series(ax, timestamp: np.ndarray, values: np.ndarray, title: str, ylabel: str) -> None:
	values = np.asarray(values, dtype=float)
	if values.ndim == 1:
		values = values[:, None]

	for index in range(values.shape[1]):
		ax.plot(timestamp, values[:, index], label=f"{ylabel} {index + 1}")

	ax.set_title(title)
	ax.set_xlabel("Timestamp [s]")
	ax.set_ylabel(ylabel)
	ax.grid(True, alpha=0.3)
	if values.shape[1] > 1:
		ax.legend(loc="best")


def plot_telemetry(csv_file: str | Path, save_path: str | Path | None = None) -> None:
	csv_path = Path(csv_file)
	timestamp, position, force, data_label = _load_telemetry_csv(csv_path)
	velocity, acceleration, jerk = _differentiate(position, timestamp)

	figure, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
	figure.suptitle(f"{data_label.capitalize()} telemetry analysis: {csv_path.name}")

	_plot_series(axes[0], timestamp, position, "Position", "pos")
	_plot_series(axes[1], timestamp, velocity, "Velocity", "vel")
	_plot_series(axes[2], timestamp, acceleration, "Acceleration", "acc")
	_plot_series(axes[3], timestamp, jerk, "Jerk", "jerk")
	_plot_series(axes[4], timestamp, force, "Force / Torque", "force")

	figure.tight_layout()
	figure.subplots_adjust(top=0.95)

	if save_path is not None:
		output_path = Path(save_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(output_path, dpi=200, bbox_inches="tight")

	plt.show()


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot telemetry CSV data.")
	parser.add_argument("csv_file", help="Path to a telemetry CSV file")
	parser.add_argument(
		"--save",
		dest="save_path",
		help="Optional output image path (for example telemetry.png)",
	)
	args = parser.parse_args()
	plot_telemetry(args.csv_file, args.save_path)


if __name__ == "__main__":
	main()