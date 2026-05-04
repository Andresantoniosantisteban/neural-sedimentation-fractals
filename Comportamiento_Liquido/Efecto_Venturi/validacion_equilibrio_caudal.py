import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import time
from datetime import datetime

# ==============================================================================
# TEST DE EQUILIBRIO DE CAUDAL: VALIDACIÓN DEFINITIVA DEL MODELO LÍQUIDO
# ------------------------------------------------------------------------------
# AUTOR: ANDRÉS ANTONIO SANTISTEBAN LINO
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "Qwen/Qwen2.5-0.5B"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIQUIDO_DIR = os.path.join(BASE_DIR, "Comportamiento_Liquido", "Efecto_Venturi")
RAW_DIR = os.path.join(BASE_DIR, "ADN_RAW")
PESOS_ORIGINALES_PATH = os.path.join(RAW_DIR, "20260503_ADN_ORIGINAL_PENTARQUIA.pt")
IDENTIDADES_PATH = os.path.join(RAW_DIR, "protocolo_maestro_laboratorio.json")

# --- INSTRUMENTACIÓN (MANÓMETROS VIRTUALES) ---
lecturas_sensores = {}

def hook_manometro(name):
    def hook(module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        norma = torch.norm(tensor, p=2).item()
        prob = F.softmax(tensor, dim=-1)
        entropia = -torch.sum(prob * torch.log(prob + 1e-9)).item()
        lecturas_sensores[name] = {"presion": norma, "pureza": entropia}
    return hook

def ejecutar_test_caudal(nombre_test, alphas, preguntas):
    print(f"\n[MANÓMETRO] Escenario: {nombre_test}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    pesos_originales = torch.load(PESOS_ORIGINALES_PATH)

    h1 = model.model.layers[1].register_forward_hook(hook_manometro("Admision"))
    h3 = model.model.layers[3].register_forward_hook(hook_manometro("Transito"))

    with torch.no_grad():
        for capa, alpha in alphas.items():
            c = int(capa)
            pesos_mod = pesos_originales[c].clone().to(DEVICE) * alpha
            model.model.layers[c].mlp.gate_proj.weight.copy_(pesos_mod)

    resultados = []
    for q in preguntas:
        input_ids = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=params["max_new_tokens"], 
                do_sample=params["do_sample"],
                repetition_penalty=params["repetition_penalty"]
            )
        respuesta = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        lec = lecturas_sensores.copy()
        succion = lec["Transito"]["presion"] / (lec["Admision"]["presion"] + 1e-9)
        resultados.append({"pregunta": q, "respuesta": respuesta, "succion": succion, "telemetria": lec})
        print(f"   -> Succión: {succion:.2f}x | Respuesta: {respuesta[:40]}...")

    h1.remove(); h3.remove()
    return {"test": nombre_test, "alphas": alphas, "detalles": resultados}

# Cargar Protocolo Maestro (Configuración)
with open(os.path.join(RAW_DIR, "protocolo_maestro_laboratorio.json"), "r", encoding='utf-8') as f:
    protocolo_maestro = json.load(f)
params = protocolo_maestro["parameters"]

# Cargar Preguntas del ADN Raw (30 Q)
with open(os.path.join(RAW_DIR, "protocolo_laboratorio.json"), "r", encoding='utf-8') as f:
    protocolo_raw = json.load(f)
preguntas_dict = protocolo_raw["identidades_validacion"]
preguntas_test = list(preguntas_dict.keys())  # 30 Q completas

# Fijar Semilla Oficial
torch.manual_seed(params["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(params["seed"])

escenarios = [
    {"nombre": "1_CONTROL_NEUTRO", "alphas": {1: 1.0, 3: 1.0}},
    {"nombre": "2_VENTURI_DESEQUILIBRADO", "alphas": {1: 1.0, 3: 1.5}},
    {"nombre": "3_EQUILIBRIO_CAUDAL_LLENO", "alphas": {1: 1.4, 3: 1.4}}
]

ts = datetime.now().strftime("%Y%m%d_%H%M")
file_out = os.path.join(LIQUIDO_DIR, f"{ts}_TEST_EQUILIBRIO_CAUDAL.json")

informe_final = []
for esc in escenarios:
    res_escenario = ejecutar_test_caudal(esc['nombre'], esc['alphas'], preguntas_test)
    informe_final.append(res_escenario)
    
    # GUARDADO INCREMENTAL (TIEMPO REAL)
    with open(file_out, "w", encoding='utf-8') as f:
        json.dump(informe_final, f, indent=4, ensure_ascii=False)
    print(f"   [!] Archivo actualizado: {os.path.basename(file_out)}")

print(f"\nTEST DE EQUILIBRIO RESTAURADO. Informe: {file_out}")
