from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import discretize_observation, epsilon_greedy_action, moving_average, save_json


N_BINS = 10
CARTPOLE_LIMITS = [(-2.4, 2.4), (-3.0, 3.0), (-0.3, 0.3), (-3.0, 3.0)]
CARTPOLE_BINS = [np.linspace(low, high, N_BINS - 1) for low, high in CARTPOLE_LIMITS]


@dataclass
class CartPoleRun:
    rewards: list[float]
    q_table: np.ndarray
    label: str


def build_cartpole_q_table() -> np.ndarray:
    return np.zeros([N_BINS] * 4 + [2], dtype=np.float64)


def train_cartpole_broken(episodes: int = 3000, seed: int = 42) -> CartPoleRun:
    """Replica exacta del agente roto de la tarea (4 errores conceptuales)."""
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    q_table = build_cartpole_q_table()

    alpha = 0.1
    gamma = 0.0
    epsilon = 1.0
    eps_min = 0.01
    eps_decay = 1.0

    rewards: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        state = discretize_observation(obs, CARTPOLE_BINS)
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            obs2, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_observation(obs2, CARTPOLE_BINS)
            done = terminated or truncated

            q_table[state][action] = alpha * (
                reward + gamma * np.max(q_table[state]) - q_table[state][action]
            )

            state = next_state
            total_reward += float(reward)
            epsilon = max(eps_min, epsilon * eps_decay)

        rewards.append(total_reward)

    env.close()
    return CartPoleRun(rewards=rewards, q_table=q_table, label="Agente roto")


def train_cartpole_fixed(episodes: int = 3000, seed: int = 42) -> CartPoleRun:
    """Q-Learning corregido con exploracion, gamma y actualizacion Bellman correcta."""
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    q_table = build_cartpole_q_table()

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    eps_min = 0.01
    eps_decay = 0.9995

    rewards: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        state = discretize_observation(obs, CARTPOLE_BINS)
        total_reward = 0.0
        done = False

        while not done:
            action = epsilon_greedy_action(q_table[state], epsilon, rng)
            obs2, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_observation(obs2, CARTPOLE_BINS)
            done = terminated or truncated

            bootstrap = 0.0 if done else float(np.max(q_table[next_state]))
            target = reward + gamma * bootstrap
            q_table[state][action] += alpha * (target - q_table[state][action])

            state = next_state
            total_reward += float(reward)
            epsilon = max(eps_min, epsilon * eps_decay)

        rewards.append(total_reward)

    env.close()
    return CartPoleRun(rewards=rewards, q_table=q_table, label="Agente corregido")


def evaluate_cartpole_agent(
    q_table: np.ndarray,
    episodes: int = 20,
    seed: int = 123,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    episode_rewards: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        state = discretize_observation(obs, CARTPOLE_BINS)
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize_observation(obs, CARTPOLE_BINS)
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
    }


CARTPOLE_ACTION_LABELS = {0: "Empujar izquierda", 1: "Empujar derecha"}
CARTPOLE_FIELDS = [
    ("posicion_carro", "Posicion del carro", -2.4, 2.4),
    ("velocidad_carro", "Velocidad del carro", -3.0, 3.0),
    ("angulo_palo", "Angulo del palo (rad)", -0.3, 0.3),
    ("velocidad_angular", "Velocidad angular del palo", -3.0, 3.0),
]


def predict_cartpole(q_table: np.ndarray, inputs: dict[str, float]) -> dict[str, object]:
    obs = np.array(
        [
            float(inputs["posicion_carro"]),
            float(inputs["velocidad_carro"]),
            float(inputs["angulo_palo"]),
            float(inputs["velocidad_angular"]),
        ],
        dtype=np.float64,
    )

    for idx, (_, _, low, high) in enumerate(CARTPOLE_FIELDS):
        if obs[idx] < low or obs[idx] > high:
            raise ValueError(f"Valor fuera de rango para {CARTPOLE_FIELDS[idx][1]}: [{low}, {high}]")

    discrete = discretize_observation(obs, CARTPOLE_BINS)
    q_values = q_table[discrete]
    action = int(np.argmax(q_values))

    return {
        "estado_continuo": [float(x) for x in obs],
        "estado_discreto": list(discrete),
        "accion": action,
        "accion_label": CARTPOLE_ACTION_LABELS[action],
        "q_izquierda": float(q_values[0]),
        "q_derecha": float(q_values[1]),
        "confianza": float(np.max(q_values) - np.min(q_values)),
        "interpretacion": (
            "El agente mantiene el palo vertical empujando hacia "
            f"{'la izquierda' if action == 0 else 'la derecha'} segun el estado actual."
        ),
    }


def run_cartpole_episode(q_table: np.ndarray, seed: int | None = None) -> dict[str, object]:
    env = gym.make("CartPole-v1", render_mode=None)
    obs, _ = env.reset(seed=seed)
    state = discretize_observation(obs, CARTPOLE_BINS)
    total_reward = 0.0
    steps: list[dict[str, object]] = []
    done = False

    while not done:
        action = int(np.argmax(q_table[state]))
        obs2, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_observation(obs2, CARTPOLE_BINS)
        done = terminated or truncated
        total_reward += float(reward)
        steps.append(
            {
                "estado": [float(x) for x in obs2],
                "estado_discreto": list(next_state),
                "accion": int(action),
                "recompensa": float(reward),
            }
        )
        state = next_state

    env.close()
    return {
        "recompensa_total": float(total_reward),
        "pasos": steps,
        "llego_meta": bool(float(total_reward) >= 195.0),
    }


def plot_cartpole_comparison(broken: CartPoleRun, fixed: CartPoleRun, output_path: Path) -> None:
    broken_ma = moving_average(broken.rewards, window=100)
    fixed_ma = moving_average(fixed.rewards, window=100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(broken_ma, label="Agente roto (media movil 100)", color="#c0392b", linewidth=2)
    ax.plot(fixed_ma, label="Agente corregido (media movil 100)", color="#1f7a4c", linewidth=2)
    ax.set_title("CartPole: curva de aprendizaje roto vs corregido")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa (media movil)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def train_and_export_part_a(artifacts_dir: Path) -> dict[str, object]:
    broken = train_cartpole_broken()
    fixed = train_cartpole_fixed()

    plot_cartpole_comparison(broken, fixed, artifacts_dir / "figures" / "cartpole_roto_vs_corregido.png")
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    np.savez(
        artifacts_dir / "models" / "cartpole_broken.npz",
        q_table=broken.q_table,
        rewards=np.asarray(broken.rewards, dtype=np.float64),
    )
    np.savez(
        artifacts_dir / "models" / "cartpole_fixed.npz",
        q_table=fixed.q_table,
        rewards=np.asarray(fixed.rewards, dtype=np.float64),
    )

    evaluation = {
        "roto": evaluate_cartpole_agent(broken.q_table),
        "corregido": evaluate_cartpole_agent(fixed.q_table),
    }
    save_json(artifacts_dir / "metrics" / "parte_a_evaluacion.json", evaluation)

    errors_doc = {
        "errores": [
            {
                "linea": "GAMMA=0.0",
                "elemento_mdp": "retorno",
                "problema": "Ignora recompensas futuras; el retorno G_t deja de propagarse.",
                "correccion": "GAMMA=0.99",
            },
            {
                "linea": "a = np.argmax(Q[s])",
                "elemento_mdp": "exploracion / politica",
                "problema": "Siempre explota sin epsilon-greedy; no visita estados-acciones nuevos.",
                "correccion": "Seleccion epsilon-greedy entre explorar y explotar.",
            },
            {
                "linea": "EPS_DECAY=1.0",
                "elemento_mdp": "exploracion",
                "problema": "Epsilon nunca decae, impidiendo converger a una politica estable.",
                "correccion": "EPS_DECAY=0.9995 con epsilon usado en la seleccion de acciones.",
            },
            {
                "linea": "np.max(Q[s]) en la actualizacion",
                "elemento_mdp": "funcion Q",
                "problema": "Bootstrap con el estado actual viola la ecuacion de Bellman de Q-Learning.",
                "correccion": "Usar np.max(Q[s2]) y 0 cuando el episodio termina.",
            },
        ]
    }
    save_json(artifacts_dir / "metrics" / "parte_a_errores.json", errors_doc)

    return {
        "broken_rewards": broken.rewards,
        "fixed_rewards": fixed.rewards,
        "evaluation": evaluation,
        "errors": errors_doc["errores"],
    }