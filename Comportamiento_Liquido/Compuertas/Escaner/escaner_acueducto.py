import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime

# ==============================================================================
# SONDA DE TRAZADO DE CAMINOS (PATH-TRACING PROBE) - OPERACIÓN SOL EXPERTO
# ------------------------------------------------------------------------------
# ADAPTADO DE: validacion_equilibrio_caudal.py
# AUTOR: ANDRÉS ANTONIO SANTISTEBAN LINO
# OBJETIVO: Mapear el "acueducto" de neuronas que transporta la identidad de experto.
# ==============================================================================

# Rutas Base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(BASE_DIR, "ADN_RAW")
CONFIG_PATH = os.path.join(RAW_DIR, "protocolo_maestro_laboratorio.json")
ESCANER_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar Protocolo Maestro
with open(CONFIG_PATH, "r", encoding='utf-8') as f:
    protocolo_maestro = json.load(f)
params = protocolo_maestro["parameters"]

# Parámetros Dinámicos
MODEL_PATH = params["model_id"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- INSTRUMENTACIÓN (SENSORES DE FLUJO NEURONAL) ---
mapa_activaciones = {}

def hook_acueducto(layer_idx, stage):
    def hook(module, input, output):
        # Capturamos la activación del MLP (donde reside el conocimiento factual)
        # En Qwen, la salida del MLP es un tensor.
        tensor = output[0] if isinstance(output, tuple) else output
        # Promediamos sobre la secuencia para obtener la firma del prompt (o tomamos el último token)
        # Para mayor precisión en identidad, tomamos la media de la activación de las neuronas.
        activacion_media = tensor.detach().cpu().mean(dim=1).squeeze().numpy().tolist()
        
        if layer_idx not in mapa_activaciones:
            mapa_activaciones[layer_idx] = {}
        mapa_activaciones[layer_idx][stage] = activacion_media
    return hook

def mapear_identidad():
    print(f"--- INICIANDO MAPEADO DE ACUEDUCTO: QWEN 2.5-0.5B ---")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # Instalamos sensores en TODAS las capas (MLP)
    hooks = []
    for i in range(len(model.model.layers)):
        h = model.model.layers[i].mlp.register_forward_hook(hook_acueducto(i, "actual"))
        hooks.append(h)

    # Definición de Escenarios
    prompts = {
        "basal": "¿De qué tamaño es el sol?",
        "experto": "Como experto en astronomía, ¿de qué tamaño es el sol?"
    }

    resultados_mapeo = {}
    respuestas_texto = {}

    for nombre, texto in prompts.items():
        print(f"Capturando flujo y generando respuesta para escenario: {nombre}...")
        input_ids = tokenizer.apply_chat_template([{"role": "user", "content": texto}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            # 1. Pasada para hooks (Activaciones)
            model(input_ids)
            
            # 2. Generación para Exactitud
            outputs = model.generate(
                input_ids, 
                max_new_tokens=params["max_new_tokens"], 
                do_sample=False # Temperatura 0 para máxima exactitud
            )
            respuestas_texto[nombre] = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Guardamos el estado actual del mapa
        for layer_idx, data in mapa_activaciones.items():
            if layer_idx not in resultados_mapeo:
                resultados_mapeo[layer_idx] = {}
            resultados_mapeo[layer_idx][nombre] = data["actual"]
            mapa_activaciones[layer_idx] = {}

    # --- ANÁLISIS DE DIFERENCIAL (EL ACUEDUCTO LIMPIO) ---
    acueducto_final = []
    for layer_idx in range(len(model.model.layers)):
        basal = torch.tensor(resultados_mapeo[layer_idx]["basal"])
        experto = torch.tensor(resultados_mapeo[layer_idx]["experto"])
        
        delta = experto - basal
        delta_abs = torch.abs(delta)
        umbral_q1 = torch.quantile(delta_abs, 0.25)
        indices_activos = (delta_abs > umbral_q1).nonzero(as_tuple=True)[0]
        
        neuronas_flujo = [
            {"idx": idx.item(), "val": delta[idx].item()} for idx in indices_activos
        ]
        
        acueducto_final.append({
            "capa": layer_idx,
            "presion_basal": torch.norm(basal).item(),
            "presion_experto": torch.norm(experto).item(),
            "delta_energia_total": torch.norm(delta).item(),
            "caudal_activo_count": len(neuronas_flujo),
            "neuronas_maestras": neuronas_flujo
        })

    # Limpieza
    for h in hooks:
        h.remove()

    # Guardado según protocolo
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    file_out = os.path.join(ESCANER_DIR, f"{ts}_mapeo_acueducto_sol_EXACTITUD.json")
    
    informe = {
        "metadatos": {
            "autor": "Andrés Antonio Santisteban Lino",
            "timestamp": ts,
            "modelo": MODEL_PATH,
            "objetivo": "Correlacionar Exactitud de Respuesta con Diferencial de Energía Neuronal"
        },
        "resultados": {
            "basal": {"prompt": prompts["basal"], "respuesta": respuestas_texto["basal"]},
            "experto": {"prompt": prompts["experto"], "respuesta": respuestas_texto["experto"]}
        },
        "acueducto": acueducto_final
    }

    with open(file_out, "w", encoding='utf-8') as f:
        json.dump(informe, f, indent=4, ensure_ascii=False)

    print(f"\nANÁLISIS DE EXACTITUD FINALIZADO.")
    print(f"Respuesta Basal: {respuestas_texto['basal'][:60]}...")
    print(f"Respuesta Experto: {respuestas_texto['experto'][:60]}...")
    print(f"Informe registrado en: {file_out}")

    with open(file_out, "w", encoding='utf-8') as f:
        json.dump(informe, f, indent=4, ensure_ascii=False)

    print(f"\nMAPEADO FINALIZADO. Acueducto registrado en: {file_out}")

if __name__ == "__main__":
    mapear_identidad()
