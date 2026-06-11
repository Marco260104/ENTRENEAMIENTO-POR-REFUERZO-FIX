from __future__ import annotations

import time
from pathlib import Path

from src.parte_a import train_and_export_part_a
from src.parte_b import train_and_export_part_b
from src.parte_c import train_and_export_part_c
from src.utils import save_json


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary: dict[str, object] = {"partes": {}}

    print("Entrenando Parte A (CartPole roto vs corregido)...")
    summary["partes"]["A"] = train_and_export_part_a(ARTIFACTS_DIR)

    print("Entrenando Parte B (MountainCar)...")
    summary["partes"]["B"] = train_and_export_part_b(ARTIFACTS_DIR)

    print("Entrenando Parte C (Inventario propio)...")
    summary["partes"]["C"] = train_and_export_part_c(ARTIFACTS_DIR)

    summary["duracion_segundos"] = round(time.time() - started, 2)
    save_json(ARTIFACTS_DIR / "metrics" / "resumen_entrenamiento.json", summary)
    print(f"Entrenamiento completo en {summary['duracion_segundos']}s")
    print(f"Artefactos guardados en: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()