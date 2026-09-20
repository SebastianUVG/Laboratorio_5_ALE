"""Compara checkpoints según el máximo esperado en bloques de cinco partidas."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sb3_contrib import QRDQN

from .configuracion import Configuracion
from .entornos import crear_entorno_vectorizado
from .evaluar import evaluar_modelo, resumen


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seleccionar el checkpoint más competitivo")
    parser.add_argument(
        "--episodios",
        type=int,
        default=5,
        help="Episodios por semilla; use 5 para replicar la competencia",
    )
    parser.add_argument(
        "--semillas",
        type=int,
        default=1,
        help="Número de bloques independientes que se evaluarán por modelo",
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=1000.0,
        help="Puntaje que se considera excelente dentro de cada bloque",
    )
    parser.add_argument(
        "--ultimos-checkpoints",
        type=int,
        default=8,
        help="Además de best_model y modelo_final, compara los N checkpoints más recientes",
    )
    parser.add_argument("--seed", type=int, default=33092)
    parser.add_argument("--dispositivo", default="auto")
    return parser.parse_args()


def maximo_esperado_de_cinco(puntajes: list[float], seed: int) -> float:
    """Bootstrap reproducible de la métrica usada el día de competencia."""
    rng = np.random.default_rng(seed)
    muestras = rng.choice(np.asarray(puntajes), size=(20_000, 5), replace=True)
    return float(muestras.max(axis=1).mean())


def candidatos(config: Configuracion, ultimos_checkpoints: int) -> list[Path]:
    checkpoints = sorted(
        config.checkpoints_dir.glob("*.zip"),
        key=lambda ruta: ruta.stat().st_mtime_ns,
    )
    rutas = checkpoints[-ultimos_checkpoints:] if ultimos_checkpoints > 0 else []
    # Incluye también respaldos manuales del mejor modelo anterior, para que
    # una continuación que empeore no desplace al agente ya validado.
    rutas.extend(config.mejor_modelo_dir.glob("*.zip"))
    modelo_final = config.modelos_dir / "modelo_final.zip"
    if modelo_final.exists():
        rutas.append(modelo_final)
    return sorted(set(rutas))


def main() -> None:
    args = argumentos()
    if args.episodios <= 0 or args.semillas <= 0:
        raise ValueError("--episodios y --semillas deben ser mayores que cero")
    config = Configuracion()
    config.crear_directorios()
    modelos = candidatos(config, args.ultimos_checkpoints)
    if not modelos:
        raise FileNotFoundError("No se encontraron modelos o checkpoints para comparar")

    resultados: list[dict[str, object]] = []
    for indice, ruta in enumerate(modelos):
        print(f"\n[{indice + 1}/{len(modelos)}] {ruta}")
        puntajes: list[float] = []
        evaluaciones_por_semilla: list[dict[str, object]] = []
        maximos_por_bloque: list[float] = []

        for numero_semilla in range(args.semillas):
            # Separación grande y determinista para producir streams de ALE
            # independientes, pero idénticos entre todos los candidatos.
            semilla = args.seed + numero_semilla * 1_000_003
            print(f"  Semilla {numero_semilla + 1}/{args.semillas}: {semilla}")
            env = crear_entorno_vectorizado(
                config,
                entrenamiento=False,
                seed=semilla,
                monitor_file=None,
            )
            try:
                modelo = QRDQN.load(
                    ruta,
                    env=env,
                    device=args.dispositivo,
                    buffer_size=1,
                    learning_starts=0,
                )
                # QRDQN.load restaura el seed de entrenamiento; se vuelve a
                # aplicar el de este bloque antes del primer reset.
                env.seed(semilla)
                puntajes_bloque = evaluar_modelo(modelo, env, args.episodios)
            finally:
                env.close()

            puntajes.extend(puntajes_bloque)
            maximo_bloque = max(puntajes_bloque)
            maximos_por_bloque.append(maximo_bloque)
            evaluaciones_por_semilla.append(
                {
                    "seed": semilla,
                    "puntajes": puntajes_bloque,
                    "promedio": float(np.mean(puntajes_bloque)),
                    "maximo": maximo_bloque,
                }
            )

        fila = {
            "modelo": str(ruta),
            "episodios_por_semilla": args.episodios,
            "numero_semillas": args.semillas,
            "umbral": args.umbral,
            "evaluaciones_por_semilla": evaluaciones_por_semilla,
            **resumen(puntajes),
            "maximo_esperado_5": maximo_esperado_de_cinco(
                puntajes, args.seed
            ),
            "promedio_maximos_por_bloque": float(np.mean(maximos_por_bloque)),
            "peor_maximo_por_bloque": float(np.min(maximos_por_bloque)),
            "probabilidad_empirica_superar_umbral": float(
                np.mean(np.asarray(maximos_por_bloque) >= args.umbral)
            ),
        }
        resultados.append(fila)

    # Prioriza cuántos bloques de cinco producen al menos un gran resultado;
    # después el máximo promedio del bloque y finalmente el retorno general.
    resultados.sort(
        key=lambda x: (
            float(x["probabilidad_empirica_superar_umbral"]),
            float(x["promedio_maximos_por_bloque"]),
            float(x["promedio"]),
        ),
        reverse=True,
    )
    ganador = Path(str(resultados[0]["modelo"]))
    destino = config.modelos_dir / "modelo_seleccionado.zip"
    shutil.copy2(ganador, destino)

    reporte = config.evaluaciones_dir / "comparacion_modelos.json"
    reporte.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModelo seleccionado: {ganador}")
    print(f"Copia para competencia: {destino}")


if __name__ == "__main__":
    main()
