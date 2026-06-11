from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import epsilon_greedy_action, moving_average, save_json


POSITION_BINS = 24
VELOCITY_BINS = 20
MOUNTAINCAR_LIMITS = [(-1.2, 0.6), (-0.07, 0.07)]
MOUNTAINCAR_BINS = [
    np.linspace(MOUNTAINCAR_LIMITS[0][0], MOUNTAINCAR_LIMITS[0][1], POSITION_BINS - 1),
    np.linspace(MOUNTAINCAR_LIMITS[1][0], MOUNTAINCAR_LIMITS[1][1], VELOCITY_BINS - 1),
]


@dataclass
class MountainCarRun:
    rewards: list[float]
    q_table: np.ndarray
    success_rate: float


def build_mountaincar_q_table() -> np.ndarray:
    return np.zeros([POSITION_BINS, VELOCITY_BINS, 3], dtype=np.float64)


def explore_mountaincar_random(episodes: int = 5, seed: int = 7) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    observation_space = str(env.observation_space)
    action_space = str(env.action_space)
    episode_rewards: list[float] = []
    traces: list[dict[str, object]] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            steps += 1

        episode_rewards.append(total_reward)
        traces.append({"episodio": episode + 1, "recompensa": float(total_reward), "pasos": steps})

    env.close()
    arr = np.asarray(episode_rewards, dtype=np.float64)
    return {
        "observation_space": observation_space,
        "action_space": action_space,
        "episodios_aleatorios": traces,
        "recompensa_promedio": float(arr.mean()),
        "recompensa_min": float(arr.min()),
        "recompensa_max": float(arr.max()),
        "q_table_shape": [POSITION_BINS, VELOCITY_BINS, 3],
        "q_table_entries": POSITION_BINS * VELOCITY_BINS * 3,
        "bins": {"posicion": POSITION_BINS, "velocidad": VELOCITY_BINS},
    }


def discretize_mountaincar(obs: np.ndarray) -> tuple[int, int]:
    clipped = np.array(
        [
            float(np.clip(obs[0], MOUNTAINCAR_LIMITS[0][0], MOUNTAINCAR_LIMITS[0][1])),
            float(np.clip(obs[1], MOUNTAINCAR_LIMITS[1][0], MOUNTAINCAR_LIMITS[1][1])),
        ],
        dtype=np.float64,
    )
    pos = int((clipped[0] + 1.2) / 1.8 * POSITION_BINS)
    vel = int((clipped[1] + 0.07) / 0.14 * VELOCITY_BINS)
    return (
        min(POSITION_BINS - 1, max(0, pos)),
        min(VELOCITY_BINS - 1, max(0, vel)),
    )


def train_mountaincar(episodes: int = 20000, seed: int = 42) -> MountainCarRun:
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    q_table = build_mountaincar_q_table()

    alpha = 0.15
    gamma = 0.99
    epsilon = 1.0
    eps_min = 0.08
    eps_decay = 0.99975

    rewards: list[float] = []
    successes = 0
    best_episode_reward = float("-inf")
    best_q = q_table.copy()

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        state = discretize_mountaincar(obs)
        total_reward = 0.0
        done = False

        while not done:
            action = epsilon_greedy_action(q_table[state], epsilon, rng)
            obs2, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_mountaincar(obs2)
            done = terminated or truncated

            bootstrap = 0.0 if done else float(np.max(q_table[next_state]))
            target = reward + gamma * bootstrap
            q_table[state][action] += alpha * (target - q_table[state][action])

            state = next_state
            total_reward += float(reward)

        rewards.append(total_reward)
        if total_reward > -110:
            successes += 1
        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
            best_q = q_table.copy()

        epsilon = max(eps_min, epsilon * eps_decay)

    env.close()
    success_rate = successes / episodes
    return MountainCarRun(rewards=rewards, q_table=best_q, success_rate=success_rate)


def evaluate_mountaincar_agent(q_table: np.ndarray, episodes: int = 20, seed: int = 99) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    episode_rewards: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        state = discretize_mountaincar(obs)
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize_mountaincar(obs)
            done = terminated or truncated
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    env.close()
    arr = np.asarray(episode_rewards, dtype=np.float64)
    return {
        "promedio": float(arr.mean()),
        "maxima": float(arr.max()),
        "minima": float(arr.min()),
        "episodios": episodes,
        "llego_meta_consistente": bool(arr.mean() > -130),
    }


MOUNTAINCAR_ACTION_LABELS = {
    0: "Acelerar a la izquierda",
    1: "Sin acelerar",
    2: "Acelerar a la derecha",
}


def predict_mountaincar(q_table: np.ndarray, posicion: float, velocidad: float) -> dict[str, object]:
    if posicion < MOUNTAINCAR_LIMITS[0][0] or posicion > MOUNTAINCAR_LIMITS[0][1]:
        raise ValueError(f"Posicion fuera de rango [{MOUNTAINCAR_LIMITS[0][0]}, {MOUNTAINCAR_LIMITS[0][1]}]")
    if velocidad < MOUNTAINCAR_LIMITS[1][0] or velocidad > MOUNTAINCAR_LIMITS[1][1]:
        raise ValueError(f"Velocidad fuera de rango [{MOUNTAINCAR_LIMITS[1][0]}, {MOUNTAINCAR_LIMITS[1][1]}]")

    obs = np.array([float(posicion), float(velocidad)], dtype=np.float64)
    discrete = discretize_mountaincar(obs)
    q_values = q_table[discrete]
    action = int(np.argmax(q_values))

    meta_posicion = 0.5
    distancia_meta = float(meta_posicion - posicion)

    return {
        "posicion": float(posicion),
        "velocidad": float(velocidad),
        "estado_discreto": list(discrete),
        "accion": action,
        "accion_label": MOUNTAINCAR_ACTION_LABELS[action],
        "q_izquierda": float(q_values[0]),
        "q_neutro": float(q_values[1]),
        "q_derecha": float(q_values[2]),
        "distancia_meta": distancia_meta,
        "interpretacion": (
            f"El auto esta a {distancia_meta:.3f} de la meta. "
            f"El agente elige {MOUNTAINCAR_ACTION_LABELS[action].lower()} para ganar impulso."
        ),
    }


def run_mountaincar_episode(q_table: np.ndarray, seed: int | None = None) -> dict[str, object]:
    env = gym.make("MountainCar-v0")
    obs, _ = env.reset(seed=seed)
    state = discretize_mountaincar(obs)
    total_reward = 0.0
    steps: list[dict[str, object]] = []
    done = False

    while not done:
        action = int(np.argmax(q_table[state]))
        obs2, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_mountaincar(obs2)
        done = terminated or truncated
        total_reward += reward
        steps.append(
            {
                "posicion": float(obs2[0]),
                "velocidad": float(obs2[1]),
                "estado_discreto": list(next_state),
                "accion": action,
                "recompensa": float(reward),
            }
        )
        state = next_state

    env.close()
    return {
        "recompensa_total": float(total_reward),
        "pasos": steps,
        "llego_meta": bool(float(total_reward) > -110),
    }


def plot_mountaincar_learning(run: MountainCarRun, output_path: Path) -> None:
    ma = moving_average(run.rewards, window=100)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ma, color="#2c4c7c", linewidth=2)
    ax.set_title("MountainCar: curva de aprendizaje (media movil 100)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def train_and_export_part_b(artifacts_dir: Path) -> dict[str, object]:
    exploration = explore_mountaincar_random()
    run = train_mountaincar()
    evaluation = evaluate_mountaincar_agent(run.q_table)

    plot_mountaincar_learning(run, artifacts_dir / "figures" / "mountaincar_curva.png")
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    np.savez(
        artifacts_dir / "models" / "mountaincar.npz",
        q_table=run.q_table,
        rewards=np.asarray(run.rewards, dtype=np.float64),
        bins_position=MOUNTAINCAR_BINS[0],
        bins_velocity=MOUNTAINCAR_BINS[1],
    )

    hyperparams = {
        "alpha": 0.15,
        "gamma": 0.99,
        "epsilon_inicial": 1.0,
        "epsilon_min": 0.08,
        "epsilon_decay": 0.99975,
        "episodios": 20000,
        "bins_posicion": POSITION_BINS,
        "bins_velocidad": VELOCITY_BINS,
        "justificacion": {
            "gamma_alto": "Recompensa -1 por paso exige valorar trayectorias cortas hacia la meta.",
            "exploracion_larga": "MountainCar es sparse-reward; requiere epsilon alto por mas tiempo.",
            "mas_bins": "Mayor granularidad en posicion/velocidad para capturar impulso en la cuesta.",
        },
    }

    save_json(artifacts_dir / "metrics" / "parte_b_exploracion.json", exploration)
    save_json(artifacts_dir / "metrics" / "parte_b_hiperparametros.json", hyperparams)
    save_json(artifacts_dir / "metrics" / "parte_b_evaluacion.json", evaluation)

    return {
        "exploration": exploration,
        "rewards": run.rewards,
        "evaluation": evaluation,
        "success_rate": run.success_rate,
        "hyperparams": hyperparams,
    }