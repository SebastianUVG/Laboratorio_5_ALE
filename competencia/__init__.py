"""Agente competitivo para ALE/SpaceInvaders-v5."""

from .configuracion import Configuracion
from .entornos import crear_entorno_vectorizado

__all__ = ["Configuracion", "crear_entorno_vectorizado"]
