"""Entrena QR-DQN y conserva checkpoints y el mejor modelo evaluado."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import gymnasium
import sb3_contrib
import stable_baselines3
import torch
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from .configuracion import Configuracion
from .entornos import crear_entorno_vectorizado


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar QR-DQN en Space Invaders")
    parser.add_argument("--pasos", type=int, default=10_000_000)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="100000 requiere aproximadamente 6 GB de RAM con 4 frames",
    )
    parser.add_argument("--seed", type=int, default=3092)
    parser.add_argument("--dispositivo", default="auto", help="auto, cuda, mps o cpu")
    parser.add_argument("--reanudar", type=Path, help="Checkpoint .zip para continuar")
    parser.add_argument(
        "--cargar-buffer",
        type=Path,
        help="Replay buffer .pkl compatible (opcional al reanudar)",
    )
    parser.add_argument(
        "--guardar-buffer",
        action="store_true",
        help="Guarda el replay buffer final; puede ocupar varios GB",
    )
    return parser.parse_args()


def guardar_manifiesto(config: Configuracion, ruta: Path) -> None:
    manifiesto = {
        "fecha_inicio": datetime.now().isoformat(timespec="seconds"),
        "configuracion": config.como_dict(),
        "versiones": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
        },
        "cuda_disponible": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = argumentos()
    config = Configuracion(
        total_timesteps=args.pasos,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )
    config.crear_directorios()
    set_random_seed(config.seed)
    guardar_manifiesto(config, config.root / "configuracion_entrenamiento.json")

    train_env = crear_entorno_vectorizado(
        config,
        entrenamiento=True,
        seed=config.seed,
        monitor_file=config.logs_dir / "entrenamiento.monitor.csv",
    )
    eval_env = crear_entorno_vectorizado(
        config,
        entrenamiento=False,
        seed=config.seed + 10_000,
        monitor_file=config.logs_dir / "evaluacion_periodica.monitor.csv",
    )

    if args.reanudar:
        model = QRDQN.load(
            args.reanudar,
            env=train_env,
            device=args.dispositivo,
            # Permite reducir el consumo de RAM también al reanudar; de otro
            # modo SB3 conservaría el tamaño almacenado dentro del checkpoint.
            buffer_size=config.buffer_size,
        )
        if args.cargar_buffer:
            model.load_replay_buffer(args.cargar_buffer)
        else:
            # Evita optimizar inmediatamente sobre un buffer recién creado y
            # casi vacío cuando se reanuda solo desde los pesos.
            transiciones_calentamiento = min(
                config.learning_starts,
                config.buffer_size,
            )
            model.learning_starts = model.num_timesteps + transiciones_calentamiento
            print(
                "Aviso: no se cargó replay buffer; se recolectarán "
                f"{transiciones_calentamiento} transiciones antes de volver a optimizar."
            )
        reset_num_timesteps = False
    else:
        model = QRDQN(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            tau=1.0,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            optimize_memory_usage=False,
            target_update_interval=config.target_update_interval,
            exploration_fraction=config.exploration_fraction,
            exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log=str(config.tensorboard_dir),
            policy_kwargs={"n_quantiles": config.n_quantiles},
            verbose=1,
            seed=config.seed,
            device=args.dispositivo,
        )
        reset_num_timesteps = True

    evaluador = EvalCallback(
        eval_env,
        best_model_save_path=str(config.mejor_modelo_dir),
        log_path=str(config.evaluaciones_dir),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
        warn=True,
    )
    checkpoints = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(config.checkpoints_dir),
        name_prefix="qrdqn_space_invaders",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList([evaluador, checkpoints]),
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(config.modelos_dir / "modelo_final")
        if args.guardar_buffer:
            model.save_replay_buffer(config.modelos_dir / "replay_buffer_final.pkl")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
