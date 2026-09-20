# Proyecto 2 — Agente competitivo para Space Invaders

Esta carpeta añade un agente **QR-DQN** sobre el proyecto del Laboratorio 5. El
entrenamiento queda deliberadamente sin ejecutar: los comandos siguientes están
preparados para que se corran en la máquina que tenga la GPU y el tiempo
disponibles.

## Por qué QR-DQN

QR-DQN aprende una distribución de retornos mediante regresión cuantílica, en
lugar de aproximar solamente el valor Q promedio. Para Atari suele ser más
robusto que DQN básico y mantiene las ventajas de replay buffer, red objetivo y
exploración ε-greedy. La implementación utilizada es la oficial de
[SB3-Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html),
con la CNN de Atari tipo Nature y 200 cuantiles.

La configuración de 10 millones de pasos, cuatro frames apilados y los
hiperparámetros base sigue de cerca el perfil Atari de
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/qrdqn.yml).
No existe garantía de un puntaje específico: RL tiene varianza entre semillas,
pero este punto de partida es sustancialmente más sólido que un DQN con valores
predeterminados o una política entrenada durante pocos pasos.

## Diseño del entorno

Todos los scripts llaman a una sola fábrica en `competencia/entornos.py`:

- Entorno objetivo: `ALE/SpaceInvaders-v5`.
- Las acciones pegajosas de v5 permanecen activas con probabilidad `0.25`.
- ALE se crea con `frameskip=1` y `AtariWrapper` repite cada acción cuatro
  frames. Esto evita aplicar salto 4 dos veces.
- Se usan las seis acciones mínimas; las doce acciones adicionales no aportan
  comportamiento útil y dificultan la exploración.
- Cada imagen se convierte a gris, se redimensiona a 84×84, se apilan cuatro
  frames y se entregan a la CNN con forma `(4, 84, 84)`.
- Durante entrenamiento, perder una vida funciona como terminal episódica y la
  recompensa se recorta a su signo. El estado real del juego continúa, como en
  el protocolo DQN de Atari.
- Durante evaluación y video se conservan todas las vidas y la recompensa sin
  recortar. Así, los valores reportados son los puntos reales del juego.

Esta separación es intencional: el clipping estabiliza la optimización, pero no
debe utilizarse para calcular el resultado de la competencia.

## Archivos

| Archivo | Uso |
|---|---|
| `configuracion.py` | Única fuente de hiperparámetros y rutas. |
| `entornos.py` | Preprocesamiento idéntico para train, evaluación y video. |
| `entrenar.py` | Entrenamiento, TensorBoard, checkpoints y mejor modelo. |
| `evaluar.py` | Protocolo greedy de cinco episodios y reporte JSON. |
| `seleccionar_modelo.py` | Compara checkpoints con el máximo esperado de cinco episodios. |
| `grabar_video.py` | Carga pesos y graba una partida completa. |
| `graficar_resultados.py` | Produce curvas PNG para el trabajo escrito. |

## Preparación

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-competencia.txt
```

Para verificar que PyTorch reconoce la GPU antes de invertir horas en entrenar:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Entrenamiento recomendado

Entrenamiento competitivo completo:

```powershell
python -m competencia.entrenar --pasos 10000000 --dispositivo cuda
```

El buffer de 100,000 transiciones puede consumir alrededor de 6 GB de RAM por
las observaciones apiladas. Si la máquina tiene menos de 16 GB de RAM, use:

```powershell
python -m competencia.entrenar --pasos 10000000 --buffer-size 50000 --dispositivo cuda
```

Para una primera iteración corta destinada a descubrir errores de instalación
o estimar velocidad —no debe considerarse el modelo final—:

```powershell
python -m competencia.entrenar --pasos 100000 --buffer-size 50000 --dispositivo cuda
```

Los artefactos aparecen en `resultados_competencia/`:

- `modelos/mejor/best_model.zip`: mayor recompensa promedio periódica.
- `modelos/checkpoints/`: pesos cada 250,000 pasos.
- `modelos/modelo_final.zip`: estado al terminar todos los pasos.
- `evaluaciones/evaluations.npz`: evaluaciones para elaborar gráficas.
- `logs/tensorboard/`: recompensa, pérdida, exploración y valores Q.

Visualización durante el entrenamiento:

```powershell
tensorboard --logdir resultados_competencia/logs/tensorboard
```

### Reanudar

Sin replay buffer se conservan los pesos y se recolectan 50,000 experiencias
nuevas antes de volver a actualizar la red:

```powershell
python -m competencia.entrenar --pasos 2000000 --reanudar resultados_competencia/modelos/checkpoints/qrdqn_space_invaders_5000000_steps.zip --dispositivo cuda
```

Para una continuación más fiel, la corrida inicial debe usar
`--guardar-buffer`; el archivo puede ocupar varios GB:

```powershell
python -m competencia.entrenar --pasos 2000000 --reanudar resultados_competencia/modelos/modelo_final.zip --cargar-buffer resultados_competencia/modelos/replay_buffer_final.pkl --dispositivo cuda
```

## Selección y evaluación final

No se recomienda presentar automáticamente el último checkpoint. Para simular
semillas desconocidas, se comparan el mejor, el final y los ocho checkpoints más
recientes en varios bloques independientes de cinco episodios. El selector
prioriza la frecuencia con que cada bloque supera 1000 puntos, luego el promedio
de sus máximos y finalmente el promedio general:

```powershell
python -m competencia.seleccionar_modelo --episodios 5 --semillas 8 --umbral 1000 --dispositivo cuda
```

El ganador queda copiado como
`resultados_competencia/modelos/modelo_seleccionado.zip`. Cada candidato habrá
sido medido en 40 episodios repartidos como lo estará la competencia, no como una
única secuencia larga. Luego se ejecuta el protocolo exacto de cinco partidas:

```powershell
python -m competencia.evaluar --modelo resultados_competencia/modelos/modelo_seleccionado.zip --episodios 5 --dispositivo cuda
```

El JSON resultante conserva los cinco puntajes, promedio, desviación, mínimo y
máximo de competencia. Para evitar escoger un modelo por suerte, conviene tomar
la decisión con 20 episodios y reservar otra semilla para los cinco finales.

Para observar escenarios distintos en cada ejecución puede usarse
`--seed-aleatoria`. El script imprime y guarda la semilla generada, por lo que
una corrida interesante se puede repetir después con `--seed NUMERO`:

```powershell
python -m competencia.evaluar --modelo resultados_competencia/modelos/modelo_seleccionado.zip --episodios 5 --seed-aleatoria
```

## Video y gráficas

```powershell
python -m competencia.grabar_video --modelo resultados_competencia/modelos/modelo_seleccionado.zip --episodios 5 --dispositivo cuda
python -m competencia.graficar_resultados
```

El grabador conserva un MP4 por episodio e indica la ruta del que obtuvo mayor
puntaje. Usa exactamente el mismo preprocesamiento de evaluación. `RecordVideo`
se ubica dentro del wrapper de frame skipping para conservar los frames internos
del juego, y el entorno se cierra en `finally` para finalizar correctamente el
MP4. Al cargar un modelo solo para evaluar o grabar, el replay buffer se reduce a
una transición porque no participa en las predicciones; esto evita reservar más
de 1 GB de RAM innecesariamente.

## Iteraciones sugeridas para documentar

1. **Baseline:** 100,000–500,000 pasos para validar la canalización. No se espera
   todavía un buen desempeño.
2. **QR-DQN intermedio:** 2–5 millones de pasos. Analizar la curva de evaluación,
   pérdida cuantílica y ε.
3. **Modelo final:** 10 millones de pasos; comparar `best_model`, el último
   modelo y checkpoints tardíos.
4. Si la curva continúa subiendo al final, reanudar 2–5 millones de pasos. Si se
   estanca o cae, conservar el checkpoint seleccionado y no sobreentrenar.

Para cada corrida deben anotarse semilla, pasos, cambios, recompensa promedio y
máxima, problemas y solución. No debe reportarse el reward recortado como si
fuera puntuación del juego.
