"""Infraestructura reutilizable para agentes de Gymnasium y Atari ALE.

El modulo no entrena modelos. Proporciona agentes de referencia, ejecucion de
episodios y grabacion de videos mediante la API moderna de Gymnasium.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np

Observacion: TypeAlias = Any
Agente: TypeAlias = Callable[[Observacion, gym.Env], Any]


def crear_entorno(
    nombre_entorno: str,
    video_folder: str | Path | None = None,
    episode_trigger: Callable[[int], bool] | None = None,
    name_prefix: str = "rl-video",
    **kwargs: Any,
) -> gym.Env:
    """Crea un entorno y, opcionalmente, lo prepara para grabar episodios.

    Los argumentos adicionales se envian directamente a ``gym.make``; por eso
    la funcion sirve para ALE y para otros entornos de Gymnasium. Al solicitar
    video se fuerza ``render_mode='rgb_array'``, requisito de ``RecordVideo``.
    """
    if not nombre_entorno:
        raise ValueError("nombre_entorno no puede estar vacio")

    if nombre_entorno.startswith("ALE/"):
        try:
            import ale_py
        except ImportError as exc:
            raise ImportError(
                "Para usar entornos ALE instale las dependencias con "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        gym.register_envs(ale_py)

    if video_folder is not None:
        render_mode = kwargs.get("render_mode")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(
                "RecordVideo necesita render_mode='rgb_array'; "
                f"se recibio {render_mode!r}."
            )
        kwargs["render_mode"] = "rgb_array"

    env = gym.make(nombre_entorno, **kwargs)

    if video_folder is not None:
        carpeta = Path(video_folder)
        carpeta.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(carpeta),
            episode_trigger=episode_trigger or (lambda _episodio: True),
            name_prefix=name_prefix,
            disable_logger=True,
        )

    return env


def agente_aleatorio(observation: Observacion, env: gym.Env) -> Any:
    """Devuelve una accion aleatoria valida (baseline sin entrenamiento)."""
    del observation
    return env.action_space.sample()


def _nombres_acciones(env: gym.Env) -> list[str]:
    """Obtiene los significados ALE sin suponer indices fijos."""
    try:
        return list(env.unwrapped.get_action_meanings())
    except (AttributeError, NotImplementedError):
        return []


def _imagen_2d(observation: Observacion) -> np.ndarray | None:
    """Normaliza RGB, gris o frame stacks a una imagen bidimensional."""
    imagen = np.asarray(observation)
    if imagen.ndim == 4:  # FrameStackObservation: usar el frame mas reciente
        imagen = imagen[-1]
    if imagen.ndim == 3 and imagen.shape[-1] in (1, 3, 4):
        imagen = imagen[..., :3].max(axis=-1)
    if imagen.ndim != 2:
        return None
    return imagen


def agente_regla_simple(observation: Observacion, env: gym.Env) -> Any:
    """Heuristica visual sencilla para Space Invaders.

    Localiza aproximadamente el canon en el cuarto inferior y los pixeles de
    invasores en la zona superior. Se mueve hacia el centro de masa horizontal
    de esos pixeles y dispara simultaneamente. Si la observacion no es una
    imagen o el entorno no expone acciones ALE, usa una accion aleatoria, de
    modo que la funcion falle de forma segura en otros entornos.
    """
    nombres = _nombres_acciones(env)
    imagen = _imagen_2d(observation)
    if imagen is None or not nombres:
        return env.action_space.sample()

    alto, ancho = imagen.shape
    mascara = imagen > 20

    # El canon aparece cerca del borde inferior; se ignoran marcador y suelo.
    zona_canon = mascara[int(alto * 0.78) : int(alto * 0.94)]
    columnas_canon = np.flatnonzero(zona_canon.any(axis=0))
    x_canon = float(np.median(columnas_canon)) if columnas_canon.size else ancho / 2

    # Evita HUD, escudos y canon. Las filas inferiores pesan mas para priorizar
    # a los invasores que representan una amenaza inmediata.
    y0, y1 = int(alto * 0.10), int(alto * 0.68)
    zona_invasores = mascara[y0:y1]
    pesos_y = np.linspace(0.25, 1.0, zona_invasores.shape[0])[:, None]
    pesos_x = (zona_invasores * pesos_y).sum(axis=0)
    x_objetivo = (
        float(np.average(np.arange(ancho), weights=pesos_x))
        if pesos_x.sum() > 0
        else ancho / 2
    )

    margen = max(3.0, ancho * 0.025)
    if x_objetivo > x_canon + margen:
        preferidas = ("RIGHTFIRE", "RIGHT", "FIRE", "NOOP")
    elif x_objetivo < x_canon - margen:
        preferidas = ("LEFTFIRE", "LEFT", "FIRE", "NOOP")
    else:
        preferidas = ("FIRE", "NOOP")

    for nombre in preferidas:
        if nombre in nombres:
            return nombres.index(nombre)
    return env.action_space.sample()


def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: Agente,
    max_steps: int = 10_000,
) -> dict[str, int | float | bool]:
    """Ejecuta un episodio y devuelve longitud, retorno y causa de cierre."""
    if max_steps <= 0:
        raise ValueError("max_steps debe ser mayor que cero")

    observation, _info = env.reset()
    recompensa_total = 0.0
    terminated = truncated = False

    for pasos in range(1, max_steps + 1):
        accion = funcion_agente(observation, env)
        observation, reward, terminated, truncated, _info = env.step(accion)
        recompensa_total += float(reward)
        if terminated or truncated:
            break

    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "limite_alcanzado": not (terminated or truncated),
    }


def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: Agente,
    video_folder: str | Path,
    name_prefix: str,
    n_episodios: int = 1,
    *,
    max_steps: int = 10_000,
    **kwargs_entorno: Any,
) -> tuple[list[str], list[dict[str, int | float | bool]]]:
    """Graba episodios completos y retorna videos nuevos junto con metricas."""
    if n_episodios <= 0:
        raise ValueError("n_episodios debe ser mayor que cero")

    carpeta = Path(video_folder)
    carpeta.mkdir(parents=True, exist_ok=True)
    # RecordVideo puede reutilizar un nombre (por ejemplo ``episode-0``) al
    # repetir una corrida. Guardamos una huella para detectar archivos nuevos
    # y tambien archivos sobrescritos durante esta llamada.
    existentes = {
        p.resolve(): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in carpeta.glob("*.mp4")
    }

    env = crear_entorno(
        nombre_entorno,
        video_folder=carpeta,
        episode_trigger=lambda episodio: episodio < n_episodios,
        name_prefix=name_prefix,
        **kwargs_entorno,
    )
    metricas: list[dict[str, int | float | bool]] = []
    try:
        for _ in range(n_episodios):
            metricas.append(ejecutar_episodio(env, funcion_agente, max_steps))
    finally:
        # RecordVideo termina de codificar el ultimo archivo durante close().
        env.close()

    videos = []
    for archivo in carpeta.glob("*.mp4"):
        ruta = archivo.resolve()
        huella_actual = (archivo.stat().st_mtime_ns, archivo.stat().st_size)
        if archivo.name.startswith(name_prefix) and existentes.get(ruta) != huella_actual:
            videos.append(str(ruta))
    videos.sort()
    return videos, metricas


__all__ = [
    "crear_entorno",
    "agente_aleatorio",
    "agente_regla_simple",
    "ejecutar_episodio",
    "generar_video_agente",
]
