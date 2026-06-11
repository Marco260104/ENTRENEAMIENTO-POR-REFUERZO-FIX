from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from src.parte_a import predict_cartpole, run_cartpole_episode
from src.parte_b import predict_mountaincar, run_mountaincar_episode
from src.parte_c import predict_inventory, run_inventory_episode
from src.utils import load_json, moving_average


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
FIGURES = ARTIFACTS / "figures"
METRICS = ARTIFACTS / "metrics"

app = Flask(__name__)


def image_to_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def load_npz_q(path: Path, key: str = "q_table") -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(path)
    return data[key]


def ensure_models_loaded() -> dict[str, str]:
    missing = []
    required = [
        MODELS / "cartpole_fixed.npz",
        MODELS / "cartpole_broken.npz",
        MODELS / "mountaincar.npz",
        MODELS / "inventario.npz",
    ]
    for item in required:
        if not item.exists():
            missing.append(item.name)
    return {"ready": len(missing) == 0, "missing": ", ".join(missing)}


@app.route("/")
def home():
    status = ensure_models_loaded()

    parte_a_eval = load_json(METRICS / "parte_a_evaluacion.json") if (METRICS / "parte_a_evaluacion.json").exists() else {}
    parte_b_eval = load_json(METRICS / "parte_b_evaluacion.json") if (METRICS / "parte_b_evaluacion.json").exists() else {}
    parte_b_explore = load_json(METRICS / "parte_b_exploracion.json") if (METRICS / "parte_b_exploracion.json").exists() else {}
    parte_b_hparams = load_json(METRICS / "parte_b_hiperparametros.json") if (METRICS / "parte_b_hiperparametros.json").exists() else {}
    parte_c_mdp = load_json(METRICS / "parte_c_mdp.json") if (METRICS / "parte_c_mdp.json").exists() else {}

    broken_npz = np.load(MODELS / "cartpole_broken.npz") if (MODELS / "cartpole_broken.npz").exists() else None
    fixed_npz = np.load(MODELS / "cartpole_fixed.npz") if (MODELS / "cartpole_fixed.npz").exists() else None
    mountain_npz = np.load(MODELS / "mountaincar.npz") if (MODELS / "mountaincar.npz").exists() else None
    inventory_npz = np.load(MODELS / "inventario.npz") if (MODELS / "inventario.npz").exists() else None

    cartpole_curve = None
    if broken_npz is not None and fixed_npz is not None:
        cartpole_curve = {
            "roto": moving_average(broken_npz["rewards"], 100).tolist(),
            "corregido": moving_average(fixed_npz["rewards"], 100).tolist(),
        }

    mountain_curve = None
    if mountain_npz is not None:
        mountain_curve = moving_average(mountain_npz["rewards"], 100).tolist()

    inventory_curve = None
    if inventory_npz is not None:
        inventory_curve = moving_average(inventory_npz["rewards"], 100).tolist()

    return render_template(
        "index.html",
        status=status,
        parte_a_eval=parte_a_eval,
        parte_b_eval=parte_b_eval,
        parte_b_explore=parte_b_explore,
        parte_b_hparams=parte_b_hparams,
        parte_c_mdp=parte_c_mdp,
        cartpole_img=image_to_base64(FIGURES / "cartpole_roto_vs_corregido.png"),
        mountain_img=image_to_base64(FIGURES / "mountaincar_curva.png"),
        inventory_img=image_to_base64(FIGURES / "inventario_curva.png"),
        cartpole_curve=cartpole_curve,
        mountain_curve=mountain_curve,
        inventory_curve=inventory_curve,
        q_shapes={
            "cartpole": list(np.load(MODELS / "cartpole_fixed.npz")["q_table"].shape) if (MODELS / "cartpole_fixed.npz").exists() else [],
            "mountaincar": list(np.load(MODELS / "mountaincar.npz")["q_table"].shape) if (MODELS / "mountaincar.npz").exists() else [],
            "inventario": list(np.load(MODELS / "inventario.npz")["q_table"].shape) if (MODELS / "inventario.npz").exists() else [],
        },
    )


@app.route("/api/predict/cartpole/<agent_type>", methods=["POST"])
def api_predict_cartpole(agent_type: str):
    path = MODELS / ("cartpole_fixed.npz" if agent_type == "corregido" else "cartpole_broken.npz")
    q_table = load_npz_q(path)
    if q_table is None:
        return jsonify({"error": "Modelo no encontrado. Ejecuta train_all.py"}), 404

    try:
        data = request.get_json(force=True)
        result = predict_cartpole(
            q_table,
            {
                "posicion_carro": float(data["posicion_carro"]),
                "velocidad_carro": float(data["velocidad_carro"]),
                "angulo_palo": float(data["angulo_palo"]),
                "velocidad_angular": float(data["velocidad_angular"]),
            },
        )
        return jsonify(result)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict/mountaincar", methods=["POST"])
def api_predict_mountaincar():
    q_table = load_npz_q(MODELS / "mountaincar.npz")
    if q_table is None:
        return jsonify({"error": "Modelo no encontrado. Ejecuta train_all.py"}), 404

    try:
        data = request.get_json(force=True)
        result = predict_mountaincar(q_table, float(data["posicion"]), float(data["velocidad"]))
        return jsonify(result)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict/inventario", methods=["POST"])
def api_predict_inventario():
    q_table = load_npz_q(MODELS / "inventario.npz")
    if q_table is None:
        return jsonify({"error": "Modelo no encontrado. Ejecuta train_all.py"}), 404

    try:
        data = request.get_json(force=True)
        result = predict_inventory(
            q_table,
            stock=float(data["stock"]),
            demand_level=int(data["demanda_nivel"]),
            days_delivery=int(data["dias_entrega"]),
        )
        return jsonify(result)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/run/cartpole/<agent_type>", methods=["POST"])
def api_run_cartpole(agent_type: str):
    path = MODELS / ("cartpole_fixed.npz" if agent_type == "corregido" else "cartpole_broken.npz")
    q_table = load_npz_q(path)
    if q_table is None:
        return jsonify({"error": "Modelo no encontrado. Ejecuta train_all.py"}), 404

    seed = request.json.get("seed") if request.is_json else None
    result = run_cartpole_episode(q_table, seed=seed)
    return jsonify(result)


@app.route("/api/run/mountaincar", methods=["POST"])
def api_run_mountaincar():
    q_table = load_npz_q(MODELS / "mountaincar.npz")
    if q_table is None:
        return jsonify({"error": "Modelo no encontrado. Ejecuta train_all.py"}), 404

    seed = request.json.get("seed") if request.is_json else None
    result = run_mountaincar_episode(q_table, seed=seed)
    return jsonify(result)


@app.route("/api/run/inventario", methods=["POST"])
def api_run_inventario():
    q_table = load_npz_q(MODELS / "inventario.npz")
    if q_table is None:
        return jsonify({"error": "Modelo no encontrado. Ejecuta train_all.py"}), 404

    seed = request.json.get("seed") if request.is_json else None
    result = run_inventory_episode(q_table, seed=seed)
    return jsonify(result)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    import subprocess
    import sys

    subprocess.run([sys.executable, str(ROOT / "train_all.py")], check=True)
    return jsonify({"ok": True, "message": "Entrenamiento completo finalizado."})


if __name__ == "__main__":
    if not ensure_models_loaded()["ready"]:
        print("Modelos no encontrados. Ejecutando entrenamiento inicial...")
        from train_all import main as train_main

        train_main()
    app.run(debug=False, port=5000)