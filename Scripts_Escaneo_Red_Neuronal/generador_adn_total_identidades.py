# =============================================================================
# generador_adn_total_identidades.py
# Autor: Andres Antonio Santisteban Lino
# Fecha de reconstruccion: 2026-05-05
# Descripcion: Generador reconstruido por ingenieria inversa del archivo
#              ADN_TOTAL_IDENTIDADES.json. Escanea el 100% de las neuronas
#              (24 capas x 4864 = 116,736) para los tokens del SUJETO de
#              cada pregunta, registrando score crudo (s), impacto
#              estructural (im), ranking global cross-layer (r) y
#              porcentaje relativo (p).
# Entrada: ADN_RAW/protocolo_maestro_laboratorio.json (parametros del modelo)
#          ADN_RAW/protocolo_laboratorio.json (preguntas y sujetos)
# Salida:  ADN_RAW/YYYYMMDD_HHMM_ADN_TOTAL_IDENTIDADES.json
# =============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import numpy as np
from datetime import datetime
import sys

# --- CONFIGURACION DE RUTAS ---
# Todas las rutas son relativas a la raiz del proyecto,
# independientemente de donde se ejecute este script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ADN_RAW_DIR = os.path.join(PROJECT_ROOT, 'ADN_RAW')
PROTOCOLO_MAESTRO = os.path.join(ADN_RAW_DIR, 'protocolo_maestro_laboratorio.json')
PROTOCOLO_PREGUNTAS = os.path.join(ADN_RAW_DIR, 'protocolo_laboratorio.json')
MONITOR_PATH = os.path.join(ADN_RAW_DIR, 'MONITOR_ESCANEO.txt')


def obtener_timestamp():
    """Devuelve marca de tiempo legible para registros internos."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def obtener_prefijo_archivo():
    """Devuelve prefijo de archivo segun protocolo de nomenclatura."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def actualizar_monitor(sujeto, token, progreso):
    """Escribe el estado actual del escaneo en el archivo de monitor."""
    with open(MONITOR_PATH, 'w', encoding='utf-8') as f:
        f.write(f"--- MONITOR DE ESCANEO TOTAL ---\n")
        f.write(f"Ultimo Pulso: {obtener_timestamp()}\n")
        f.write(f"Sujeto Actual: {sujeto.upper()}\n")
        f.write(f"Token: {token}\n")
        f.write(f"Progreso: {progreso}\n")
        f.write(f"Estado: TRABAJANDO...\n")


def identificar_tokens_sujeto(tokenizer, pregunta, sujeto):
    """
    Identifica los indices de los tokens del sujeto dentro del prompt.
    
    Logica:
    1. Tokeniza " sujeto" (con espacio, como aparece en el prompt)
    2. Si el sujeto tiene mas de 1 sub-token, descarta el primero
       (que lleva el prefijo de espacio)
    3. Busca esos sub-tokens en la secuencia completa del prompt
    
    Returns:
        Lista de tuplas (indice_en_prompt, string_del_token)
    """
    tokens_prompt = tokenizer.convert_ids_to_tokens(
        tokenizer(pregunta)['input_ids']
    )
    
    # Tokenizar el sujeto tal como aparece en el prompt (con espacio previo)
    tokens_sujeto = tokenizer.convert_ids_to_tokens(
        tokenizer(" " + sujeto)['input_ids']
    )
    
    # Si tiene mas de 1 sub-token, descartar el primero
    if len(tokens_sujeto) > 1:
        tokens_objetivo = tokens_sujeto[1:]
    else:
        tokens_objetivo = tokens_sujeto
    
    # Buscar los tokens objetivo en la secuencia del prompt
    resultado = []
    for objetivo in tokens_objetivo:
        for idx, t in enumerate(tokens_prompt):
            if t == objetivo and idx not in [r[0] for r in resultado]:
                resultado.append((idx, t))
                break
    
    return resultado


def ejecutar_escaneo(limite_preguntas=None):
    """
    Ejecuta el escaneo total de identidades neuronales.
    
    Para cada pregunta del protocolo, identifica los tokens del sujeto
    y para cada uno recorre las 24 capas x 4864 neuronas, calculando:
      - s: activacion cruda de gate_proj (float16)
      - im: impacto estructural = s * norma_columna_down_proj (float32)
      - r: ranking global descendente por impacto
      - p: porcentaje relativo al maximo impacto del token
    
    Args:
        limite_preguntas: Si se especifica, solo procesa las primeras N preguntas.
    """
    print(f"[{obtener_timestamp()}] Iniciando Escaneo Total de Identidades...")
    print(f"[{obtener_timestamp()}] Protocolo Maestro: {PROTOCOLO_MAESTRO}")
    print(f"[{obtener_timestamp()}] Protocolo Preguntas: {PROTOCOLO_PREGUNTAS}")

    # --- Verificacion de archivos de entrada ---
    if not os.path.exists(PROTOCOLO_MAESTRO):
        print(f"ERROR: No se encuentra {PROTOCOLO_MAESTRO}")
        return None
    if not os.path.exists(PROTOCOLO_PREGUNTAS):
        print(f"ERROR: No se encuentra {PROTOCOLO_PREGUNTAS}")
        return None

    # --- Carga de configuracion maestra ---
    with open(PROTOCOLO_MAESTRO, 'r', encoding='utf-8') as f:
        maestro = json.load(f)
    MODEL_ID = maestro['parameters']['model_id']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{obtener_timestamp()}] Modelo: {MODEL_ID} | Dispositivo: {DEVICE}")

    # --- Carga de preguntas ---
    with open(PROTOCOLO_PREGUNTAS, 'r', encoding='utf-8') as f:
        protocolo = json.load(f)
    base_preguntas = protocolo['identidades_validacion']

    if limite_preguntas:
        preguntas_items = list(base_preguntas.items())[:limite_preguntas]
        print(f"[{obtener_timestamp()}] Modo prueba: {limite_preguntas} preguntas")
    else:
        preguntas_items = list(base_preguntas.items())

    total_preguntas = len(preguntas_items)
    print(f"[{obtener_timestamp()}] Preguntas a procesar: {total_preguntas}")

    # --- Carga del modelo ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)
    print(f"[{obtener_timestamp()}] Modelo cargado en {DEVICE}.")

    # --- Hooks para capturar activaciones de gate_proj ---
    activaciones_temp = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            activaciones_temp[layer_idx] = output.detach().cpu()
        return hook

    num_capas = model.config.num_hidden_layers
    for i in range(num_capas):
        model.model.layers[i].mlp.gate_proj.register_forward_hook(hook_fn(i))

    # --- Pre-calculo de normas de down_proj por columna ---
    print(f"[{obtener_timestamp()}] Pre-calculando Topografia de Pesos en {DEVICE}...")
    normas_capas = {}
    with torch.no_grad():
        for i in range(num_capas):
            weights = model.model.layers[i].mlp.down_proj.weight
            normas_capas[i] = torch.norm(weights, dim=0).cpu().numpy()

    intermediate_size = normas_capas[0].shape[0]
    total_neuronas = num_capas * intermediate_size
    print(f"[{obtener_timestamp()}] Capas: {num_capas} | Neuronas/capa: {intermediate_size} | Total: {total_neuronas}")

    # --- Escaneo principal ---
    resultado_total = {}

    for idx_pregunta, (pregunta, metadata) in enumerate(preguntas_items, 1):
        sujeto = metadata['sujeto']
        print(f"[{obtener_timestamp()}] [{idx_pregunta}/{total_preguntas}] Mapeando: {sujeto.upper()}...")

        # Tokenizar el prompt completo y ejecutar forward pass
        inputs = tokenizer(pregunta, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            model(**inputs)

        # Identificar los tokens del sujeto dentro del prompt
        tokens_sujeto = identificar_tokens_sujeto(tokenizer, pregunta, sujeto)
        print(f"  Tokens del sujeto: {[t[1] for t in tokens_sujeto]}")

        datos_tokens = []
        num_tokens_sujeto = len(tokens_sujeto)

        for i_count, (t_idx, token_str) in enumerate(tokens_sujeto, 1):
            actualizar_monitor(sujeto, token_str, f"{i_count}/{num_tokens_sujeto}")

            # Recopilar TODAS las neuronas de TODAS las capas
            todas_neuronas = []
            for capa_idx in range(num_capas):
                vals = activaciones_temp[capa_idx][0, t_idx].to(torch.float32).numpy()
                col_norms = normas_capas[capa_idx]
                impactos = vals * col_norms

                for neurona_idx in range(intermediate_size):
                    todas_neuronas.append({
                        "c": capa_idx,
                        "i": int(neurona_idx),
                        "s": float(vals[neurona_idx]),
                        "im": float(impactos[neurona_idx])
                    })

            # Ordenar por impacto descendente (positivo a negativo)
            todas_neuronas.sort(key=lambda x: x['im'], reverse=True)

            # Asignar ranking y porcentaje relativo
            max_im = todas_neuronas[0]['im'] if todas_neuronas else 1.0
            for r, n in enumerate(todas_neuronas, 1):
                n['r'] = r
                n['p'] = (n['im'] / max_im) * 100.0

            datos_tokens.append({
                "token": token_str,
                "mapa_completo": todas_neuronas
            })

        resultado_total[pregunta] = {
            "sujeto": sujeto,
            "timestamp": obtener_timestamp(),
            "analisis_tokens": datos_tokens
        }

    # --- Escritura del archivo de salida ---
    prefijo = obtener_prefijo_archivo()
    nombre_salida = f"{prefijo}_ADN_TOTAL_IDENTIDADES.json"
    output_path = os.path.join(ADN_RAW_DIR, nombre_salida)

    print(f"[{obtener_timestamp()}] Escribiendo: {nombre_salida}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resultado_total, f, indent=2)

    tamano_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[{obtener_timestamp()}] Escaneo Total Finalizado.")
    print(f"Archivo: {nombre_salida} ({tamano_mb:.1f} MB)")
    print(f"Ruta: {output_path}")

    return output_path


if __name__ == "__main__":
    # Sin argumentos: ejecuta las 30 preguntas completas
    # Con argumento numerico: modo prueba con N preguntas
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ejecutar_escaneo(limite_preguntas=limite)
