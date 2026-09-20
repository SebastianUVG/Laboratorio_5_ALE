"""Evaluación greedy con recompensa original y protocolo de cinco episodios."""

from __future__ import annotations

import argparse
import json
import secrets
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import QRDQN

from .configuracion import Configuracion
from .entornos import crear_entorno_vectorizado


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluar un QR-DQN entrenado")
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path("resultados_competencia/modelos/mejor/best_model.zip"),
    )
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13092)
    parser.add_argument(
        "--seed-aleatoria",
        action="store_true",
        help="Genera una semilla distinta y reporta el valor utilizado",
    )
    parser.add_argument("--dispositivo", default="auto")
    parser.add_argument("--salida", type=Path)
    return parser.parse_args()


def evaluar_modelo(
    modelo: QRDQN,
    env,
    n_episodios: int,
) -> list[float]:
    puntajes: list[float] = []
    observation = env.reset()
    retorno = 0.0

    while len(puntajes) < n_episodios:
        action, _state = modelo.predict(observation, deterministic=True)
        observation, rewards, dones, _infos = env.step(action)
        retorno += float(rewards[0])
        if bool(dones[0]):
            puntajes.append(retorno)
            print(f"Episodio {len(puntajes)}: {retorno:.1f}")
            retorno = 0.0
    return puntajes


def resumen(puntajes: list[float]) -> dict[str, object]:
    valores = np.asarray(puntajes, dtype=np.float64)
    return {
        "puntajes": puntajes,
        "promedio": float(valores.mean()),
        "desviacion_estandar": float(valores.std()),
        "minimo": float(valores.min()),
        "maximo_competencia": float(valores.max()),
    }


def main() -> None:
    args = argumentos()
    if args.episodios <= 0:
        raise ValueError("--episodios debe ser mayor que cero")

    config = Configuracion()
    config.crear_directorios()
    salida = args.salida or config.evaluaciones_dir / "evaluacion_final.json"
    semilla = secrets.randbelow(2**31) if args.seed_aleatoria else args.seed
    print(f"Semilla utilizada: {semilla}")
    env = crear_entorno_vectorizado(
        config,
        entrenamiento=False,
        seed=semilla,
        monitor_file=config.logs_dir / "evaluacion_final.monitor.csv",
    )
    try:
        # Para inferencia no se utiliza replay buffer. Sobrescribir su tamaño
        # evita reservar más de 1 GB de RAM al cargar un modelo entrenado.
        modelo = QRDQN.load(
            args.modelo,
            env=env,
            device=args.dispositivo,
            buffer_size=1,
            learning_starts=0,
        )
        # QRDQN.load restaura la semilla guardada durante entrenamiento. Se
        # aplica nuevamente la semilla de esta evaluación antes del reset.
        env.seed(semilla)
        puntajes = evaluar_modelo(modelo, env, args.episodios)
    finally:
        env.close()

    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": str(args.modelo.resolve()),
        "seed_inicial": semilla,
        "entorno": config.env_id,
        **resumen(puntajes),
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
