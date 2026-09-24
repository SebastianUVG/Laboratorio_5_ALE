"""Genera las curvas del informe desde los registros existentes."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .configuracion import Configuracion


def escalares_tensorboard(config: Configuracion, etiqueta: str) -> tuple[np.ndarray, np.ndarray]:
    """Combina las distintas sesiones y conserva el ultimo valor de cada paso."""
    valores_por_paso: dict[int, float] = {}
    archivos = sorted(
        config.tensorboard_dir.rglob("events.out.tfevents.*"),
        key=lambda ruta: ruta.stat().st_mtime_ns,
    )
    for archivo in archivos:
        acumulador = EventAccumulator(str(archivo))
        acumulador.Reload()
        if etiqueta not in acumulador.Tags().get("scalars", []):
            continue
        for evento in acumulador.Scalars(etiqueta):
            valores_por_paso[int(evento.step)] = float(evento.value)
    if not valores_por_paso:
        raise ValueError(f"TensorBoard no contiene la etiqueta {etiqueta!r}")
    pasos = np.asarray(sorted(valores_por_paso), dtype=np.int64)
    valores = np.asarray([valores_por_paso[paso] for paso in pasos], dtype=np.float64)
    return pasos, valores


def curva_evaluacion(config: Configuracion) -> None:
    pasos, media = escalares_tensorboard(config, "eval/mean_reward")

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        pasos,
        media,
        marker="o",
        markersize=2.5,
        label="Media de 10 episodios",
        color="navy",
    )
    plt.xlabel("Pasos de entrenamiento")
    plt.ylabel("Recompensa real promedio")
    plt.title("QR-DQN - evaluacion greedy periodica")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.evaluaciones_dir / "curva_evaluacion.png", dpi=180)
    plt.close()


def curva_entrenamiento(config: Configuracion, ventana: int = 100) -> None:
    ruta = config.logs_dir / "entrenamiento.monitor.csv"
    with ruta.open(encoding="utf-8") as archivo:
        filas = csv.DictReader(linea for linea in archivo if not linea.startswith("#"))
        recompensas = np.asarray([float(fila["r"]) for fila in filas])
    if recompensas.size == 0:
        raise ValueError("El monitor todavia no contiene episodios completos")

    ventana_real = min(ventana, recompensas.size)
    suavizada = np.convolve(
        recompensas, np.ones(ventana_real) / ventana_real, mode="valid"
    )
    plt.figure(figsize=(8, 4.5))
    plt.plot(recompensas, alpha=0.18, color="gray", label="Por episodio")
    plt.plot(
        np.arange(ventana_real - 1, recompensas.size),
        suavizada,
        color="darkgreen",
        label=f"Media movil ({ventana_real})",
    )
    plt.xlabel("Episodio completo")
    plt.ylabel("Recompensa real")
    plt.title("QR-DQN - recompensa durante entrenamiento")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.evaluaciones_dir / "curva_entrenamiento.png", dpi=180)
    plt.close()


def curva_perdida(config: Configuracion, ventana: int = 25) -> None:
    pasos, perdida = escalares_tensorboard(config, "train/loss")
    ventana_real = min(ventana, perdida.size)
    suavizada = np.convolve(
        perdida, np.ones(ventana_real) / ventana_real, mode="valid"
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(pasos, perdida, alpha=0.18, color="gray", label="Perdida registrada")
    plt.plot(
        pasos[ventana_real - 1 :],
        suavizada,
        color="darkred",
        label=f"Media movil ({ventana_real})",
    )
    plt.xlabel("Pasos de entrenamiento")
    plt.ylabel("Pérdida Huber cuantílica")
    plt.title("QR-DQN - pérdida de entrenamiento")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.evaluaciones_dir / "curva_perdida.png", dpi=180)
    plt.close()


def curva_exploracion(config: Configuracion) -> None:
    pasos, epsilon = escalares_tensorboard(config, "rollout/exploration_rate")

    plt.figure(figsize=(8, 4.5))
    plt.plot(pasos, epsilon, color="darkorange")
    plt.xlabel("Pasos de entrenamiento")
    plt.ylabel("Epsilon")
    plt.title("QR-DQN - exploracion epsilon-greedy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(config.evaluaciones_dir / "curva_exploracion.png", dpi=180)
    plt.close()


def main() -> None:
    config = Configuracion()
    config.crear_directorios()
    curva_evaluacion(config)
    curva_entrenamiento(config)
    curva_perdida(config)
    curva_exploracion(config)
    print(f"Graficas guardadas en {config.evaluaciones_dir.resolve()}")


if __name__ == "__main__":
    main()
