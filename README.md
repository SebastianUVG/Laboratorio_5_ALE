# Laboratorio 5 — Agentes en ALE: Space Invaders

Entrega reproducible para **CC3092 - Deep Learning y Sistemas Inteligentes**.
Incluye la investigacion, un modulo reutilizable, pruebas automaticas y la
generacion de un video de un episodio completo.

## Instalacion

Se recomienda Python 3.11 o 3.12 y un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Uso

Abra y ejecute todas las celdas de `laboratorio_5_ale.ipynb`. Los archivos MP4
se crean en `entregables/videos/`; la ultima celda muestra las rutas, los pasos
y el retorno de cada episodio. `env.close()` se ejecuta incluso si ocurre una
excepcion, para no dejar incompleto el video.

Para validar el modulo independientemente:

```powershell
python -m pytest -q
```

La primera instalacion de `ale-py` incluye las ROM admitidas por el paquete
actual. No se entrena ningun modelo: los dos agentes incluidos son baselines.

