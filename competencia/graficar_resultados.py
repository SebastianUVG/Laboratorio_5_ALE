"""Genera las curvas requeridas por el informe desde logs ya existentes."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .configuracion import Configuracion


def curva_evaluacion(config: Configuracion) -> None:
    ruta = config.evaluaciones_dir / "evaluations.npz"
    datos = np.load(ruta)
    pasos = datos["timesteps"]
    recompensas = datos["results"]
    media = recompensas.mean(axis=1)
    desviacion = recompensas.std(axis=1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(pasos, media, label="Promedio de evaluación", color="navy")
    plt.fill_between(pasos, media - desviacion, media + desviacion, alpha=0.2)
    plt.xlabel("Pasos de entrenamiento")
    plt.ylabel("Recompensa real")
    plt.title("QR-DQN — evaluación periódica")
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
        raise ValueError("El monitor todavía no contiene episodios completos")

    ventana_real = min(ventana, recompensas.size)
    suavizada = np.convolve(
        recompensas, np.ones(ventana_real) / ventana_real, mode="valid"
    )
    plt.figure(figsize=(8, 4.5))
    plt.plot(recompensas, alpha=0.2, color="gray", label="Por episodio")
    plt.plot(
        np.arange(ventana_real - 1, recompensas.size),
        suavizada,
        color="darkgreen",
        label=f"Media móvil ({ventana_real})",
    )
    plt.xlabel("Episodio completo")
    plt.ylabel("Recompensa real")
    plt.title("QR-DQN — curva de entrenamiento")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.evaluaciones_dir / "curva_entrenamiento.png", dpi=180)
    plt.close()


def main() -> None:
    config = Configuracion()
    config.crear_directorios()
    curva_evaluacion(config)
    curva_entrenamiento(config)
    print(f"Gráficas guardadas en {config.evaluaciones_dir.resolve()}")


if __name__ == "__main__":
    main()
