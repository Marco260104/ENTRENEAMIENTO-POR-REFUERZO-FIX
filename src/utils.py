from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def moving_average(values: list[float] | np.ndarray, window: int = 100) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.size < window:
        return np.cumsum(arr) / np.arange(1, arr.size + 1)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="valid")


def epsilon_greedy_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q_values.shape[-1]))
    return int(np.argmax(q_values))


def discretize_observation(obs: np.ndarray, bins: list[np.ndarray]) -> tuple[int, ...]:
    return tuple(int(np.digitize(obs[i], bins[i])) for i in range(len(bins)))


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))