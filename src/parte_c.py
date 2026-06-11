from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import epsilon_greedy_action, moving_average, save_json


STOCK_BINS = 12
DEMAND_BINS = 8
LEAD_BINS = 5
N_ACTIONS = 3
MAX_STEPS = 30


class InventoryEnv:
    """
    Entorno MDP de gestion de inventario para una tienda.

    Estado (discretizado):
      - nivel de stock
      - demanda esperada del dia
      - dias hasta la proxima entrega pendiente

    Acciones:
      0 = no pedir
      1 = pedido pequeno (+8 unidades, costo fijo 12)
      2 = pedido grande (+20 unidades, costo fijo 25)

    Recompensa:
      -(costo_almacenamiento + penalizacion_quiebre + costo_pedido)
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.stock = 0.0
        self.pending_delivery = 0
        self.pending_units = 0
        self.demand_level = 0

    def _sample_demand_level(self) -> int:
        return int(self.rng.integers(0, DEMAND_BINS))

    def _demand_from_level(self, level: int) -> int:
        return int(3 + level * 2)

    def _discretize_state(self) -> tuple[int, int, int]:
        stock_bin = int(np.clip(self.stock // 3, 0, STOCK_BINS - 1))
        lead_bin = int(np.clip(self.pending_delivery, 0, LEAD_BINS - 1))
        return stock_bin, self.demand_level, lead_bin

    def set_state_from_inputs(self, stock: float, demand_level: int, days_delivery: int) -> tuple[int, int, int]:
        self.step_count = 0
        self.stock = float(max(0.0, stock))
        self.demand_level = int(np.clip(demand_level, 0, DEMAND_BINS - 1))
        self.pending_delivery = int(np.clip(days_delivery, 0, LEAD_BINS - 1))
        self.pending_units = 0
        return self._discretize_state()

    def reset(self) -> tuple[int, int, int]:
        self.step_count = 0
        self.stock = float(self.rng.integers(8, 28))
        self.pending_delivery = 0
        self.pending_units = 0
        self.demand_level = self._sample_demand_level()
        return self._discretize_state()

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool]:
        if action not in (0, 1, 2):
            raise ValueError("Accion invalida. Usa 0, 1 o 2.")

        order_cost = 0.0
        if action == 1 and self.pending_delivery == 0:
            self.pending_units = 8
            self.pending_delivery = 2
            order_cost = 12.0
        elif action == 2 and self.pending_delivery == 0:
            self.pending_units = 20
            self.pending_delivery = 3
            order_cost = 25.0

        demand = self._demand_from_level(self.demand_level)
        sold = min(self.stock, demand)
        stockout = demand - sold

        holding_cost = 0.45 * self.stock
        stockout_penalty = 6.0 * stockout
        reward = -(holding_cost + stockout_penalty + order_cost)

        self.stock = max(0.0, self.stock - sold)

        if self.pending_delivery > 0:
            self.pending_delivery -= 1
            if self.pending_delivery == 0:
                self.stock += self.pending_units
                self.pending_units = 0

        self.demand_level = self._sample_demand_level()
        self.step_count += 1
        done = self.step_count >= MAX_STEPS

        return self._discretize_state(), float(reward), done


def build_inventory_q_table() -> np.ndarray:
    return np.zeros([STOCK_BINS, DEMAND_BINS, LEAD_BINS, N_ACTIONS], dtype=np.float64)


@dataclass
class InventoryRun:
    rewards: list[float]
    q_table: np.ndarray


def train_inventory(episodes: int = 4000, seed: int = 42) -> InventoryRun:
    rng = np.random.default_rng(seed)
    q_table = build_inventory_q_table()

    alpha = 0.15
    gamma = 0.95
    epsilon = 1.0
    eps_min = 0.05
    eps_decay = 0.9994

    rewards: list[float] = []

    for _ in range(episodes):
        env = InventoryEnv(seed=int(rng.integers(0, 1_000_000)))
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = epsilon_greedy_action(q_table[state], epsilon, rng)
            next_state, reward, done = env.step(action)

            bootstrap = 0.0 if done else float(np.max(q_table[next_state]))
            target = reward + gamma * bootstrap
            q_table[state][action] += alpha * (target - q_table[state][action])

            state = next_state
            total_reward += reward
            epsilon = max(eps_min, epsilon * eps_decay)

        rewards.append(total_reward)

    return InventoryRun(rewards=rewards, q_table=q_table)


INVENTORY_ACTION_LABELS = {
    0: "No pedir",
    1: "Pedido pequeno (+8 uds)",
    2: "Pedido grande (+20 uds)",
}


def predict_inventory(
    q_table: np.ndarray,
    stock: float,
    demand_level: int,
    days_delivery: int,
) -> dict[str, object]:
    if stock < 0 or stock > 36:
        raise ValueError("Stock debe estar entre 0 y 36 unidades.")
    if demand_level < 0 or demand_level > DEMAND_BINS - 1:
        raise ValueError(f"Nivel de demanda debe estar entre 0 y {DEMAND_BINS - 1}.")
    if days_delivery < 0 or days_delivery > LEAD_BINS - 1:
        raise ValueError(f"Dias hasta entrega debe estar entre 0 y {LEAD_BINS - 1}.")

    env = InventoryEnv(seed=0)
    discrete = env.set_state_from_inputs(stock, demand_level, days_delivery)
    q_values = q_table[discrete]
    action = int(np.argmax(q_values))

    demand_units = env._demand_from_level(demand_level)
    stockout_risk = "Alto" if stock < demand_units else ("Medio" if stock < demand_units + 5 else "Bajo")

    sim_env = InventoryEnv(seed=0)
    sim_env.set_state_from_inputs(stock, demand_level, days_delivery)
    _, reward_if_action, _ = sim_env.step(action)

    return {
        "entrada": {
            "stock_unidades": float(stock),
            "demanda_nivel": int(demand_level),
            "demanda_unidades_estimadas": demand_units,
            "dias_hasta_entrega": int(days_delivery),
        },
        "estado_discreto": list(discrete),
        "accion": action,
        "accion_label": INVENTORY_ACTION_LABELS[action],
        "q_no_pedir": float(q_values[0]),
        "q_pedido_pequeno": float(q_values[1]),
        "q_pedido_grande": float(q_values[2]),
        "riesgo_quiebre": stockout_risk,
        "recompensa_estimada_siguiente_paso": float(reward_if_action),
        "interpretacion": (
            f"Con {stock:.0f} unidades y demanda estimada de {demand_units}, "
            f"el agente recomienda: {INVENTORY_ACTION_LABELS[action]}."
        ),
    }


def run_inventory_episode(q_table: np.ndarray, seed: int | None = None) -> dict[str, object]:
    env = InventoryEnv(seed=seed)
    state = env.reset()
    total_reward = 0.0
    steps: list[dict[str, object]] = []
    done = False

    action_labels = {0: "No pedir", 1: "Pedido pequeno", 2: "Pedido grande"}

    while not done:
        action = int(np.argmax(q_table[state]))
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps.append(
            {
                "estado": {
                    "stock": float(env.stock),
                    "demanda_nivel": int(env.demand_level),
                    "dias_entrega": int(env.pending_delivery),
                    "estado_discreto": list(next_state),
                },
                "accion": action,
                "accion_label": action_labels[action],
                "recompensa": float(reward),
            }
        )
        state = next_state

    return {"recompensa_total": float(total_reward), "pasos": steps}


def plot_inventory_learning(run: InventoryRun, output_path: Path) -> None:
    ma = moving_average(run.rewards, window=100)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ma, color="#8a4b12", linewidth=2)
    ax.set_title("Inventario: curva de aprendizaje (media movil 100)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa acumulada")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def train_and_export_part_c(artifacts_dir: Path) -> dict[str, object]:
    run = train_inventory()
    plot_inventory_learning(run, artifacts_dir / "figures" / "inventario_curva.png")
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    np.savez(
        artifacts_dir / "models" / "inventario.npz",
        q_table=run.q_table,
        rewards=np.asarray(run.rewards, dtype=np.float64),
    )

    mdp_doc = {
        "S": f"Stock x Demanda x Lead time ({STOCK_BINS}x{DEMAND_BINS}x{LEAD_BINS})",
        "A": "0=no pedir, 1=pedido pequeno, 2=pedido grande",
        "R": "Negativo de costos de almacenamiento, quiebre y pedido",
        "gamma": 0.95,
        "estado_documentado": ["stock_bin", "demand_level", "lead_time_bin"],
        "acciones_minimo": 3,
    }
    save_json(artifacts_dir / "metrics" / "parte_c_mdp.json", mdp_doc)

    return {"rewards": run.rewards, "mdp": mdp_doc}