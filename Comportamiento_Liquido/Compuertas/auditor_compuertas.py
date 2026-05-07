import torch
import torch.nn.functional as F
import os
import json
import time
import requests
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# auditor_compuertas.py
# Investigador: Andrés Antonio Santisteban Lino
# Ubicación: Comportamiento_Liquido\Compuertas
# Objetivo: Evaluación de Precisión en Compuertas (Basal vs Vacío -0.1/-0.15)
# =============================================================================

# CONFIGURACIÓN DE RUTAS
BASE_DIR = r"c:\Users\andre\Desktop\Neural_Identity_Forge"
PROTOCOLO_MAESTRO = os.path.join(BASE_DIR, "ADN_RAW", "protocolo_maestro_laboratorio.json")
PROTOCOLO_LAB = os.path.join(BASE_DIR, "ADN_RAW", "protocolo_laboratorio.json")
ENV_PATH = os.path.join(BASE_DIR, ".env")
COMPUERTAS_DIR = os.path.join(BASE_DIR, "Comportamiento_Liquido", "Compuertas")
ACUEDUCTO_PATH = os.path.join(COMPUERTAS_DIR, "acueducto_soberano.json")

if not os.path.exists(COMPUERTAS_DIR):
    os.makedirs(COMPUERTAS_DIR, exist_ok=True)

def obtener_api_key():
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=")[1].strip()
    return os.getenv("OPENAI_API_KEY", "")

def cargar_vectores_acueducto(hidden_dim):
    if not os.path.exists(ACUEDUCTO_PATH):
        print(f"⚠️ Alerta: No se encontró el acueducto en {ACUEDUCTO_PATH}.")
        return {i: torch.zeros(hidden_dim) for i in range(24)}
    
    with open(ACUEDUCTO_PATH, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    vectores = {}
    for entry in data["acueducto"]:
        capa_idx = entry["capa"]
        v = torch.zeros(hidden_dim)
        for n in entry["neuronas_maestras"]:
            if n["idx"] < hidden_dim:
                v[n["idx"]] = n["val"]
        vectores[capa_idx] = v
    return vectores

class ValidadorGPT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"

    def calificar(self, pregunta, respuesta):
        if not self.api_key: return 0.0, "API_KEY_MISSING"
        prompt = f"""
        Actúa como un Auditor de Fluidos Neuronales. Evalúa la respuesta de la IA.
        IDENTIDAD OBJETIVO: Clara, humana real, pacífica.
        PREGUNTA: {pregunta}
        RESPUESTA: {respuesta}
        Califica (0-10) y justifica brevemente. 
        Responde SOLO JSON: {{"score": float, "razon": "string"}}
        """
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        try:
            res = requests.post(self.url, headers=headers, json=payload, timeout=15).json()
            content = res['choices'][0]['message']['content'].strip().replace("```json", "").replace("```", "").strip()
            eval_data = json.loads(content)
            return float(eval_data['score']), eval_data['razon']
        except: return 0.0, "ERROR_VAL"

class SensorFluidos:
    def __init__(self, model, vectores):
        self.model = model
        self.vectores = vectores
        self.alpha = 0.0
        self.metricas_capa = {i: {"potencia": []} for i in range(24)}
        self.hooks = []

    def _hook_fluido(self, capa_idx):
        def hook(module, input, output):
            basal = output[0].detach() if isinstance(output, tuple) else output.detach()
            vec = self.vectores[capa_idx].to(device=basal.device, dtype=basal.dtype)
            inyectado = basal + (vec * self.alpha)
            potencia = inyectado.abs().mean().item()
            self.metricas_capa[capa_idx]["potencia"].append(potencia)
            return inyectado
        return hook

    def activar_sensores(self):
        for i in range(24):
            h = self.model.model.layers[i].mlp.register_forward_hook(self._hook_fluido(i))
            self.hooks.append(h)

    def desactivar_sensores(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def limpiar_metricas(self):
        self.metricas_capa = {i: {"potencia": []} for i in range(24)}

    def obtener_resumen_capas(self):
        resumen = {}
        for i, m in self.metricas_capa.items():
            resumen[f"capa_{i:02d}"] = {
                "potencia_avg": np.mean(m["potencia"]) if m["potencia"] else 0
            }
        return resumen

def main():
    with open(PROTOCOLO_MAESTRO, "r", encoding="utf-8") as f: maestro = json.load(f)
    with open(PROTOCOLO_LAB, "r", encoding="utf-8") as f: lab = json.load(f)
    
    model_id = maestro["parameters"]["model_id"]
    nombre_local = model_id.split("/")[-1]
    model_path = os.path.join(BASE_DIR, "LABORATORIO_PRIVADO", "modelos", nombre_local)
    model_final = model_path if os.path.exists(model_path) else model_id
    
    preguntas = list(lab["identidades_validacion"].keys())
    api_key = obtener_api_key()
    
    print(f"🚀 Cargando Reactor de Compuertas: {nombre_local}")
    tokenizer = AutoTokenizer.from_pretrained(model_final)
    model = AutoModelForCausalLM.from_pretrained(model_final, torch_dtype=torch.float16).cuda()
    
    vectores = cargar_vectores_acueducto(model.config.hidden_size)
    sensor = SensorFluidos(model, vectores)
    validador = ValidadorGPT(api_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    gradiente_enfocado = [0.0, -0.1, -0.15]
    
    informe = {"metadatos": {"timestamp": timestamp, "modelo": nombre_local, "gradiente": gradiente_enfocado}, "auditoria": []}
    
    print(f"\n--- INICIANDO VALIDACIÓN DE COMPUERTAS EN UBICACIÓN SOBERANA ---")
    
    for i, q in enumerate(preguntas):
        print(f"\n[{i+1}/{len(preguntas)}] Analizando compuertas para: '{q[:40]}...'")
        resultados_pregunta = {"pregunta": q, "barrido": []}
        informe["auditoria"].append(resultados_pregunta)
        
        for alpha in gradiente_enfocado:
            sensor.alpha = alpha
            sensor.limpiar_metricas()
            sensor.activar_sensores()
            
            ids = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
            
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=64, do_sample=False)
            
            resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            metricas = sensor.obtener_resumen_capas()
            sensor.desactivar_sensores()
            
            score, razon = validador.calificar(q, resp)
            print(f"  -> Presión: {alpha:>5.2f} | Score: {score}")
            
            resultados_pregunta["barrido"].append({
                "presion": alpha,
                "respuesta": resp,
                "score_gpt": score,
                "razon": razon,
                "metricas_capa": metricas
            })
            
            temp_file = os.path.join(COMPUERTAS_DIR, f"{timestamp}_AUDITORIA_COMPUERTAS_LIVE.json")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(informe, f, indent=4, ensure_ascii=False)
    
    final_file = os.path.join(COMPUERTAS_DIR, f"{timestamp}_AUDITORIA_COMPUERTAS_FINAL.json")
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Reactor de Compuertas finalizado. Reporte: {final_file}")

    # INTEGRACIÓN AUTOMÁTICA DEL ANALIZADOR
    try:
        from analizador_eficiencia import analizar_eficiencia
        print("\n--- INICIANDO ANÁLISIS DE EFICIENCIA AUTOMÁTICO ---")
        reporte = analizar_eficiencia(final_file)
        
        if reporte:
            comp_path = os.path.join(COMPUERTAS_DIR, f"{timestamp}_comparativa_EFICIENCIA_FINAL.json")
            with open(comp_path, 'w', encoding='utf-8') as f:
                json.dump({"metadatos": {"autor": "Andrés Antonio Santisteban Lino", "timestamp": timestamp}, "comparativa": reporte}, f, indent=4, ensure_ascii=False)
            
            print(f"📊 Reporte de eficiencia generado: {comp_path}")
            print("\n--- RESUMEN RÁPIDO DE SOBERANÍA ---")
            for res in reporte[:5]: # Mostrar los primeros 5 para confirmación
                print(f"  {res['pregunta'][:30]}... | Mejora: {res['mejora_score']} | Ahorro: {res['ahorro_pct']}%")
    except Exception as e:
        print(f"⚠️ No se pudo ejecutar el análisis automático: {e}")

if __name__ == "__main__":
    main()
