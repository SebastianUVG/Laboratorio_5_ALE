"""Configuración central: entrenamiento, evaluación y video usan lo mismo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Configuracion:
    # Entorno. Se usa frameskip=1 en ALE porque AtariWrapper aplica skip=4.
    env_id: str = "ALE/SpaceInvaders-v5"
    frameskip_ale: int = 1
    frameskip_agente: int = 4
    repeat_action_probability: float = 0.25
    full_action_space: bool = False
    screen_size: int = 84
    frame_stack: int = 4
    noop_max: int = 30

    # QR-DQN. Valores cercanos a los baselines Atari de RL Baselines3 Zoo.
    total_timesteps: int = 10_000_000
    learning_rate: float = 5e-5
    buffer_size: int = 100_000
    learning_starts: int = 50_000
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.025
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    n_quantiles: int = 200
    max_grad_norm: float = 10.0

    # Seguimiento y selección de modelo.
    eval_freq: int = 100_000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 250_000
    seed: int = 3092

    root: Path = Path("resultados_competencia")

    @property
    def modelos_dir(self) -> Path:
        return self.root / "modelos"

    @property
    def mejor_modelo_dir(self) -> Path:
        return self.modelos_dir / "mejor"

    @property
    def checkpoints_dir(self) -> Path:
        return self.modelos_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def tensorboard_dir(self) -> Path:
        return self.logs_dir / "tensorboard"

    @property
    def evaluaciones_dir(self) -> Path:
        return self.root / "evaluaciones"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    def crear_directorios(self) -> None:
        for directorio in (
            self.modelos_dir,
            self.mejor_modelo_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.tensorboard_dir,
            self.evaluaciones_dir,
            self.videos_dir,
        ):
            directorio.mkdir(parents=True, exist_ok=True)

    def como_dict(self) -> dict[str, Any]:
        datos = asdict(self)
        datos["root"] = str(self.root)
        return datos
