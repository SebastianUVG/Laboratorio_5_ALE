"""Construcción única del entorno para evitar diferencias train/eval/video."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from .configuracion import Configuracion


gym.register_envs(ale_py)


def _fabrica_entorno(
    config: Configuracion,
    *,
    entrenamiento: bool,
    seed: int,
    monitor_file: str | Path | None,
    video_folder: str | Path | None,
    video_prefix: str,
    n_video_episodes: int | None,
) -> Callable[[], gym.Env]:
    def _crear() -> gym.Env:
        render_mode = "rgb_array" if video_folder is not None else None
        env = gym.make(
            config.env_id,
            frameskip=config.frameskip_ale,
            repeat_action_probability=config.repeat_action_probability,
            full_action_space=config.full_action_space,
            render_mode=render_mode,
        )

        # RecordVideo debe envolver el ALE base para capturar los cuatro frames
        # internos y no solamente la observación procesada que recibe el agente.
        if video_folder is not None:
            carpeta = Path(video_folder)
            carpeta.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(carpeta),
                episode_trigger=(
                    (lambda _episodio: True)
                    if n_video_episodes is None
                    else (lambda episodio: episodio < n_video_episodes)
                ),
                name_prefix=video_prefix,
                disable_logger=True,
            )

        # Monitor se coloca antes de EpisodicLifeEnv para registrar partidas
        # completas y recompensa real, aunque entrenamiento trate cada vida
        # como terminal para facilitar la estimación de valor.
        ruta_monitor = Path(monitor_file) if monitor_file is not None else None
        env = Monitor(
            env,
            filename=str(ruta_monitor) if ruta_monitor is not None else None,
            # Una corrida nueva necesita encabezado; al reanudar se anexa al
            # CSV existente sin borrar el historial anterior.
            override_existing=ruta_monitor is None or not ruta_monitor.exists(),
        )
        env = AtariWrapper(
            env,
            noop_max=config.noop_max,
            frame_skip=config.frameskip_agente,
            screen_size=config.screen_size,
            terminal_on_life_loss=entrenamiento,
            clip_reward=entrenamiento,
            action_repeat_probability=0.0,  # ALE v5 ya aplica 0.25.
        )
        env.action_space.seed(seed)
        return env

    return _crear


def crear_entorno_vectorizado(
    config: Configuracion,
    *,
    entrenamiento: bool,
    seed: int | None = None,
    monitor_file: str | Path | None = None,
    video_folder: str | Path | None = None,
    video_prefix: str = "space-invaders-qrdqn",
    n_video_episodes: int | None = None,
) -> VecEnv:
    """Crea ALE preprocesado a (84, 84, 4), listo para una CnnPolicy.

    Entrenamiento usa vidas episódicas y reward clipping. Evaluación/video
    conservan todas las vidas y la recompensa original para medir el puntaje.
    """
    semilla = config.seed if seed is None else seed
    env = DummyVecEnv(
        [
            _fabrica_entorno(
                config,
                entrenamiento=entrenamiento,
                seed=semilla,
                monitor_file=monitor_file,
                video_folder=video_folder,
                video_prefix=video_prefix,
                n_video_episodes=n_video_episodes,
            )
        ]
    )
    env.seed(semilla)
    env = VecFrameStack(env, n_stack=config.frame_stack, channels_order="last")
    # CnnPolicy recibe explícitamente CxHxW: (4, 84, 84). Así evitamos que el
    # modelo añada un wrapper implícito distinto al cargar los pesos.
    return VecTransposeImage(env)
