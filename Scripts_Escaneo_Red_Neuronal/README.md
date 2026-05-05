# Scripts de Escaneo de Red Neuronal

**Autor:** Andres Antonio Santisteban Lino

Coleccion de scripts para el escaneo y mapeo completo de la actividad neuronal
del modelo Qwen2.5-0.5B-Instruct.

## Scripts Disponibles

### generador_adn_total_identidades.py
Escanea el 100% de las neuronas (24 capas x 4,864 = 116,736) para los tokens
del sujeto de cada pregunta del protocolo de laboratorio.

**Entrada** (desde `ADN_RAW/`):
- `protocolo_maestro_laboratorio.json` - Parametros del modelo
- `protocolo_laboratorio.json` - 30 preguntas y sujetos de validacion

**Salida** (hacia `ADN_RAW/`):
- `YYYYMMDD_HHMM_ADN_TOTAL_IDENTIDADES.json` - Mapa completo con timestamp

**Uso:**
```bash
# Ejecutar completo (30 preguntas)
python generador_adn_total_identidades.py

# Modo prueba (N preguntas)
python generador_adn_total_identidades.py 2
```

**Campos generados por neurona:**
| Campo | Descripcion |
|-------|-------------|
| `c`   | Capa (0-23) |
| `i`   | Indice de neurona (0-4863) |
| `s`   | Score crudo: activacion de gate_proj |
| `im`  | Impacto estructural: s * norma down_proj |
| `r`   | Ranking global cross-layer |
| `p`   | Porcentaje relativo al maximo impacto |

