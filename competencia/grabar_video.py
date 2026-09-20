"""Carga el agente seleccionado y graba una partida completa."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np
from sb3_contrib import QRDQN

from .configuracion import Configuracion
from .entornos import crear_entorno_vectorizado
from .evaluar import evaluar_modelo


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grabar video del agente QR-DQN")
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path("resultados_competencia/modelos/mejor/best_model.zip"),
    )
    parser.add_argument("--seed", type=int, default=23092)
    parser.add_argument(
        "--seed-aleatoria",
        action="store_true",
        help="Genera una semilla distinta y reporta el valor utilizado",
    )
    parser.add_argument(
        "--episodios",
        type=int,
        default=5,
        help="Número de partidas que se grabarán y compararán",
    )
    parser.add_argument("--dispositivo", default="auto")
    parser.add_argument("--nombre", default="agente-final")
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    if args.episodios <= 0:
        raise ValueError("--episodios debe ser mayor que cero")
    config = Configuracion()
    config.crear_directorios()
    semilla = secrets.randbelow(2**31) if args.seed_aleatoria else args.seed
    print(f"Semilla utilizada: {semilla}")
    env = crear_entorno_vectorizado(
        config,
        entrenamiento=False,
        seed=semilla,
        monitor_file=config.logs_dir / "video.monitor.csv",
        video_folder=config.videos_dir,
        video_prefix=args.nombre,
        n_video_episodes=args.episodios,
    )
    try:
        # El replay buffer no participa en predicción y consumiría memoria
        # innecesariamente en una computadora de 8 GB.
        modelo = QRDQN.load(
            args.modelo,
            env=env,
            device=args.dispositivo,
            buffer_size=1,
            learning_starts=0,
        )
        # La carga del modelo restaura su seed de entrenamiento; resembrar aquí
        # garantiza que --seed y --seed-aleatoria controlen realmente a ALE.
        env.seed(semilla)
        puntajes = evaluar_modelo(modelo, env, n_episodios=args.episodios)
    finally:
        # RecordVideo escribe y cierra el MP4 aquí.
        env.close()

    indice_mejor = int(np.argmax(puntajes))
    videos = [
        config.videos_dir / f"{args.nombre}-episode-{indice}.mp4"
        for indice in range(args.episodios)
    ]
    mejor_video = videos[indice_mejor]
    resultado = {
        "modelo": str(args.modelo.resolve()),
        "seed_inicial": semilla,
        "puntajes": puntajes,
        "mejor_episodio": indice_mejor + 1,
        "mejor_puntaje": puntajes[indice_mejor],
        "mejor_video": str(mejor_video.resolve()),
        "videos": [str(video.resolve()) for video in videos],
    }
    reporte = config.videos_dir / f"{args.nombre}-resultados.json"
    reporte.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Videos guardados en: {config.videos_dir.resolve()}")
    print(
        f"Mejor episodio: {indice_mejor + 1} | "
        f"puntaje: {puntajes[indice_mejor]:.1f}"
    )
    print(f"Video con mayor puntaje: {mejor_video.resolve()}")


if __name__ == "__main__":
    main()
