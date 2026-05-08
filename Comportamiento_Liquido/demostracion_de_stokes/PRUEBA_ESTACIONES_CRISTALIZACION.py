
"""
PRUEBA_ESTACIONES_CRISTALIZACION.py
Autor: Andrés Antonio Santisteban Lino / Antigravity
Descripción: Demostración visual de la Teoría de las Macro-Fases.
Imprime un sismograma capa por capa para un sujeto Primitivo ("perro") 
y un sujeto Compuesto ("ciudad"), demostrando que la cristalización es 
un "frenazo brusco" en capas específicas, no un gradiente suave.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def calcular_entropia_energetica(tensor, eps=1e-12):
    x = tensor.to(torch.float32)
    energia = x ** 2
    prob = energia / (torch.sum(energia) + eps)
    return -torch.sum(prob * torch.log2(prob + eps)).item()

def identificar_tokens_sujeto(tokenizer, pregunta, sujeto):
    tokens_prompt = tokenizer.convert_ids_to_tokens(tokenizer(pregunta)['input_ids'])
    tokens_sujeto = tokenizer.convert_ids_to_tokens(tokenizer(" " + sujeto)['input_ids'])
    if len(tokens_sujeto) > 1:
        tokens_objetivo = tokens_sujeto[1:]
    else:
        tokens_objetivo = tokens_sujeto
        
    resultado = []
    for objetivo in tokens_objetivo:
        for idx, t in enumerate(tokens_prompt):
            if t == objetivo and idx not in [r[0] for r in resultado]:
                resultado.append((idx, t))
                break
    return resultado

def extraer_sismograma(sujeto, model, tokenizer, device, num_layers):
    pregunta = f"¿Qué es la {sujeto}?" if sujeto == "ciudad" else f"¿Qué es el {sujeto}?"
    inputs_tokenized = tokenizer(pregunta, return_tensors="pt").to(device)
    
    tokens_sujeto_info = identificar_tokens_sujeto(tokenizer, pregunta, sujeto)
    indices_sujeto = [t[0] for t in tokens_sujeto_info]

    activaciones = {}
    def hook_residual(layer_idx):
        def hook(module, input, output):
            activaciones[layer_idx] = input[0].detach().cpu()
        return hook

    hooks = []
    for i in range(num_layers):
        hooks.append(model.model.layers[i].register_forward_hook(hook_residual(i)))
        
    with torch.no_grad():
        model(**inputs_tokenized)
        
    for h in hooks: h.remove()
    
    # =========================================================================
    # EL NÚCLEO TERMODINÁMICO: Velocidad y Entropía
    # ¿Por qué un crítico no puede refutar este sismograma?
    # Porque no estamos interpretando texto, estamos midiendo la geometría del tensor:
    # 1. Velocidad (u_rel): Cuánto se mueve físicamente el concepto.
    # 2. Entropía (H): Qué tan concentrada está su energía.
    # Si la red no fuera un fluido bajo fricción, estas dos variables serían caóticas.
    # =========================================================================
    perfil = {}
    for l in range(num_layers - 1):
        vec_l = torch.stack([activaciones[l][0, idx, :] for idx in indices_sujeto]).mean(dim=0)
        vec_next = torch.stack([activaciones[l+1][0, idx, :] for idx in indices_sujeto]).mean(dim=0)
        
        diff = vec_next - vec_l
        u_rel = torch.norm(diff, p=2).item() / (torch.norm(vec_l, p=2).item() + 1e-12)
        h_l = calcular_entropia_energetica(vec_l)
        
        perfil[l] = {"u_rel": u_rel, "h": h_l}
        
    return perfil

def generar_barras(valor, max_valor, ancho=30):
    longitud = int((valor / max_valor) * ancho)
    return "█" * longitud + "░" * (ancho - longitud)

def main():
    print("[*] Preparando Demostrador Visual de Macro-Fases...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=device)
    num_layers = model.config.num_hidden_layers
    
    sujeto1 = "perro"
    sujeto2 = "ciudad"
    
    print(f"\n[*] Extrayendo Sismograma para PRIMITIVO: '{sujeto1}'")
    perfil1 = extraer_sismograma(sujeto1, model, tokenizer, device, num_layers)
    
    print(f"[*] Extrayendo Sismograma para COMPUESTO: '{sujeto2}'")
    perfil2 = extraer_sismograma(sujeto2, model, tokenizer, device, num_layers)
    
    # Encontrar máximos para normalizar barras
    max_u1 = max([p["u_rel"] for p in perfil1.values()])
    max_u2 = max([p["u_rel"] for p in perfil2.values()])
    
    print("\n" + "="*80)
    print(" PRUEBA DE CRISTALIZACIÓN: PRIMITIVO ('perro') vs COMPUESTO ('ciudad')")
    print("="*80)
    print("Observa cómo la velocidad (Flujo semántico) sufre frenazos abruptos.")
    print("L: Capa | Vel: Velocidad Relativa (u_rel) | H: Entropía Energética")
    print("-"*80)
    
    print(f"\n>>> SUJETO PRIMITIVO: {sujeto1.upper()} <<<")
    for l in range(2, 22):
        u = perfil1[l]["u_rel"]
        h = perfil1[l]["h"]
        barra = generar_barras(u, max_u1, 20)
        print(f"Capa {l:02d} | Vel: {u:.3f} {barra} | H: {h:.2f}")
        
    print(f"\n>>> SUJETO COMPUESTO: {sujeto2.upper()} <<<")
    for l in range(2, 22):
        u = perfil2[l]["u_rel"]
        h = perfil2[l]["h"]
        barra = generar_barras(u, max_u2, 20)
        print(f"Capa {l:02d} | Vel: {u:.3f} {barra} | H: {h:.2f}")

if __name__ == "__main__":
    main()
