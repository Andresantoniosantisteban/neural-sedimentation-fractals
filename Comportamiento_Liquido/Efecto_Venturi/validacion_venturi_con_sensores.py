import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime

# ==============================================================================
# VALIDACIÓN DE VENTURI CON SENSORES (MANÓMETROS)
# ------------------------------------------------------------------------------
# Mide la caída de entropía y aumento de presión interna durante la succión.
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "Qwen/Qwen2.5-0.5B"
BASE_DIR = r"c:\Users\andre\Desktop\Neural_Identity_Forge"
LIQUIDO_DIR = os.path.join(BASE_DIR, "Entendiendo", "Estudio_Patrones", "DLA_data_sedimentaria", "Patrones_DLA", "Comportamiento_Liquido")
RAW_DIR = os.path.join(BASE_DIR, "Entendiendo", "Estudio_Patrones", "DLA_data_sedimentaria", "ADN_RAW")
PESOS_ORIGINALES_PATH = os.path.join(RAW_DIR, "20260503_ADN_ORIGINAL_PENTARQUIA.pt")
IDENTIDADES_PATH = os.path.join(BASE_DIR, "Entendiendo", "Estudio_Patrones", "DLA_data_sedimentaria", "Protocolo_Experimental", "BASE_30Q_IDENTIDADES.json")

lecturas_sensores = {}

def hook_manometro(name):
    def hook(module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        norma = torch.norm(tensor, p=2).item()
        prob = F.softmax(tensor, dim=-1)
        entropia = -torch.sum(prob * torch.log(prob + 1e-9)).item()
        lecturas_sensores[name] = {"presion_norma": norma, "pureza_entropia": entropia}
    return hook

def ejecutar_diagnostico(nombre_test, alphas, preguntas):
    print(f"\nLanzando: {nombre_test}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    pesos_originales = torch.load(PESOS_ORIGINALES_PATH)

    h1 = model.model.layers[1].register_forward_hook(hook_manometro("Capa1_Admision"))
    h3 = model.model.layers[3].register_forward_hook(hook_manometro("Capa3_Transito"))

    with torch.no_grad():
        for capa, alpha in alphas.items():
            c = int(capa)
            pesos_mod = pesos_originales[c].clone().to(DEVICE) * alpha
            model.model.layers[c].mlp.gate_proj.weight.copy_(pesos_mod)

    detalles = []
    for q in preguntas:
        input_ids = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        outputs = model.generate(input_ids, max_new_tokens=40, do_sample=False)
        respuesta = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        lec = lecturas_sensores.copy()
        succion = lec["Capa3_Transito"]["presion_norma"] / lec["Capa1_Admision"]["presion_norma"]
        detalles.append({"pregunta": q, "respuesta": respuesta, "lecturas": lec, "indice_succion": succion})
        print(f"   -> Sensor OK | Succión: {succion:.2f}x | Respuesta: {respuesta[:40]}...")

    h1.remove(); h3.remove()
    return {"test": nombre_test, "alphas": alphas, "detalles": detalles}

with open(IDENTIDADES_PATH, "r", encoding='utf-8') as f:
    base = json.load(f)
preguntas_test = list(base.keys())[:10]

resultados = [
    ejecutar_diagnostico("CONTROL_LAMINAR", {1: 1.0, 3: 1.0, 4: 1.0, 7: 1.0, 8: 1.0}, preguntas_test),
    ejecutar_diagnostico("VENTURI_FORZADO", {1: 1.0, 3: 1.5, 4: 1.0, 7: 1.0, 8: 1.0}, preguntas_test)
]

ts = datetime.now().strftime("%Y%m%d_%H%M")
file_out = os.path.join(LIQUIDO_DIR, f"{ts}_SENSORES_VENTURI.json")
with open(file_out, "w", encoding='utf-8') as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

print(f"\nDIAGNÓSTICO RESTAURADO: {file_out}")
