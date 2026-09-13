from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from agentes_ale import (
    agente_aleatorio,
    agente_regla_simple,
    crear_entorno,
    ejecutar_episodio,
    generar_video_agente,
)


def test_agente_aleatorio_retorna_accion_valida():
    env = gym.make("CartPole-v1")
    try:
        assert env.action_space.contains(agente_aleatorio(None, env))
    finally:
        env.close()


def test_ejecutar_episodio_respeta_max_steps():
    env = gym.make("CartPole-v1")
    try:
        resultado = ejecutar_episodio(env, agente_aleatorio, max_steps=3)
    finally:
        env.close()
    assert resultado["pasos"] <= 3
    assert isinstance(resultado["recompensa_total"], float)
    assert resultado["limite_alcanzado"] or resultado["terminated"] or resultado["truncated"]


def test_crear_entorno_generico_sin_video():
    env = crear_entorno("CartPole-v1")
    try:
        assert env.spec.id == "CartPole-v1"
    finally:
        env.close()


def test_validaciones():
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError):
            ejecutar_episodio(env, agente_aleatorio, max_steps=0)
    finally:
        env.close()
    with pytest.raises(ValueError):
        generar_video_agente("CartPole-v1", agente_aleatorio, "videos", "x", 0)


class _EntornoALEFalso:
    action_space = gym.spaces.Discrete(6)
    unwrapped = None

    def __init__(self):
        self.unwrapped = self

    @staticmethod
    def get_action_meanings():
        return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]


def test_agente_regla_simple_produce_accion_ale_valida():
    observacion = np.zeros((210, 160, 3), dtype=np.uint8)
    observacion[185:190, 75:85] = 255
    observacion[60:70, 110:120] = 255
    accion = agente_regla_simple(observacion, _EntornoALEFalso())
    assert accion == 4  # RIGHTFIRE
