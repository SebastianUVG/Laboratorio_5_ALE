"""Busca semillas favorables en bloques reproducibles de cinco episodios.

No entrena ni modifica el modelo: ejecuta la politica greedy y conserva en
JSON los resultados despues de cada semilla probada.
"""

from __future__ import annotations

import argparse
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import QRDQN

from .configuracion import Configuracion
from .entornos import crear_entorno_vectorizado
from .evaluar import evaluar_modelo


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buscar semillas donde todos los episodios superen un umbral"
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=Path("resultados_competencia/modelos/modelo_seleccionado.zip"),
    )
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument(
        "--umbral",
        type=float,
        default=1000.0,
        help="Recompensa minima exigida en cada episodio (por defecto: >= 1000)",
    )
    parser.add_argument(
        "--estricto",
        action="store_true",
        help="Exige recompensa > umbral en vez de >= umbral",
    )
    parser.add_argument(
        "--intentos",
        type=int,
        default=100,
        help="Numero maximo de semillas nuevas que se probaran",
    )
    parser.add_argument(
        "--seed-busqueda",
        type=int,
        help="Hace reproducible la secuencia de semillas candidatas",
    )
    parser.add_argument(
        "--continuar-al-encontrar",
        action="store_true",
        help="Prueba todos los intentos aunque ya encuentre una semilla valida",
    )
    parser.add_argument("--dispositivo", default="auto")
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("resultados_competencia/evaluaciones/busqueda_semillas.json"),
    )
    parser.add_argument(
        "--reanudar",
        action="store_true",
        help="Agrega intentos al JSON existente sin repetir sus semillas",
    )
    return parser.parse_args()


def cumple_umbral(
    puntajes: list[float],
    umbral: float,
    estricto: bool,
) -> bool:
    minimo = min(puntajes)
    return minimo > umbral if estricto else minimo >= umbral


def clave_calidad(resultado: dict[str, Any]) -> tuple[float, float, float]:
    """Prioriza consistencia: peor episodio, promedio y finalmente maximo."""
    return (
        float(resultado["minimo"]),
        float(resultado["promedio"]),
        float(resultado["maximo"]),
    )


def guardar_reporte(ruta: Path, reporte: dict[str, Any]) -> None:
    """Escritura atomica para no perder el historial si se interrumpe."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporal.replace(ruta)


def cargar_intentos(ruta: Path, reanudar: bool) -> list[dict[str, Any]]:
    if not reanudar or not ruta.exists():
        return []
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    intentos = datos.get("resultados", [])
    if not isinstance(intentos, list):
        raise ValueError(f"El archivo {ruta} no contiene una lista de resultados")
    return intentos


def construir_reporte(
    *,
    args: argparse.Namespace,
    config: Configuracion,
    semilla_busqueda: int,
    resultados: list[dict[str, Any]],
) -> dict[str, Any]:
    ordenados = sorted(resultados, key=clave_calidad, reverse=True)
    validos = [resultado for resultado in ordenados if bool(resultado["cumple_umbral"])]
    return {
        "fecha_actualizacion": datetime.now().isoformat(timespec="seconds"),
        "modelo": str(args.modelo.resolve()),
        "entorno": config.env_id,
        "episodios_por_semilla": args.episodios,
        "umbral": args.umbral,
        "comparacion": ">" if args.estricto else ">=",
        "semilla_busqueda": semilla_busqueda,
        "intentos_completados": len(resultados),
        "semillas_validas": len(validos),
        "mejor_resultado": ordenados[0] if ordenados else None,
        "mejor_resultado_valido": validos[0] if validos else None,
        "resultados": resultados,
    }


def main() -> None:
    args = argumentos()
    if args.episodios <= 0 or args.intentos <= 0:
        raise ValueError("--episodios y --intentos deben ser mayores que cero")
    if not args.modelo.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {args.modelo}")

    config = Configuracion()
    config.crear_directorios()
    resultados = cargar_intentos(args.salida, args.reanudar)
    semillas_probadas = {int(resultado["seed"]) for resultado in resultados}
    semilla_busqueda = (
        args.seed_busqueda
        if args.seed_busqueda is not None
        else secrets.randbelow(2**63)
    )
    rng = np.random.default_rng(semilla_busqueda)

    # En inferencia no se necesita reservar el replay buffer almacenado en el
    # checkpoint. El modelo tampoco necesita estar asociado a un entorno para
    # realizar predict().
    modelo = QRDQN.load(
        args.modelo,
        device=args.dispositivo,
        buffer_size=1,
        learning_starts=0,
    )

    interrumpido = False
    try:
        for numero_intento in range(1, args.intentos + 1):
            semilla = int(rng.integers(0, 2**31))
            while semilla in semillas_probadas:
                semilla = int(rng.integers(0, 2**31))
            semillas_probadas.add(semilla)

            print(f"\nIntento {numero_intento}/{args.intentos} - seed {semilla}")
            env = crear_entorno_vectorizado(
                config,
                entrenamiento=False,
                seed=semilla,
                monitor_file=None,
            )
            try:
                env.seed(semilla)
                puntajes = evaluar_modelo(modelo, env, args.episodios)
            finally:
                env.close()

            minimo = float(np.min(puntajes))
            resultado = {
                "seed": semilla,
                "puntajes": puntajes,
                "minimo": minimo,
                "promedio": float(np.mean(puntajes)),
                "maximo": float(np.max(puntajes)),
                "cumple_umbral": cumple_umbral(
                    puntajes,
                    args.umbral,
                    args.estricto,
                ),
            }
            resultados.append(resultado)
            reporte = construir_reporte(
                args=args,
                config=config,
                semilla_busqueda=semilla_busqueda,
                resultados=resultados,
            )
            guardar_reporte(args.salida, reporte)

            operador = ">" if args.estricto else ">="
            print(
                f"Minimo: {minimo:.1f} | Promedio: {resultado['promedio']:.1f} "
                f"| Objetivo: todos {operador} {args.umbral:.1f}"
            )
            if resultado["cumple_umbral"]:
                print(f"Semilla valida encontrada: {semilla}")
                if not args.continuar_al_encontrar:
                    break
    except KeyboardInterrupt:
        interrumpido = True
        print("\nBusqueda interrumpida; los intentos completados ya estan guardados.")

    reporte = construir_reporte(
        args=args,
        config=config,
        semilla_busqueda=semilla_busqueda,
        resultados=resultados,
    )
    guardar_reporte(args.salida, reporte)
    mejor = reporte["mejor_resultado"]
    print(f"\nReporte: {args.salida.resolve()}")
    if mejor is not None:
        print(
            f"Mejor seed: {mejor['seed']} | minimo: {mejor['minimo']:.1f} "
            f"| promedio: {mejor['promedio']:.1f} | puntajes: {mejor['puntajes']}"
        )
    if interrumpido:
        return


if __name__ == "__main__":
    main()
