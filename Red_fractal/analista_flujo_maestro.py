# analista_flujo_maestro.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import math
from datetime import datetime
import matplotlib.pyplot as plt

# CONFIGURACIÓN
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Estos parámetros se ajustan según la carpeta (FRACTAL o GEOMETRICO)
FOLDER = os.getcwd()
CHIP_PATH = "" # Se define abajo
TERRENO = "" # Se define abajo

PARES = [
    ("A", "Hola"),
    ("B", "punto"),
    ("C", "saber")
]

class AtomoGenerico(nn.Module):
    def __init__(self, dim, rank, layers, max_seq=5):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, dim))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "A": nn.Linear(dim, rank, bias=False),
                "B": nn.Linear(rank, dim, bias=False)
            }) for _ in range(layers)
        ])
    def forward(self, x):
        seq_len = x.size(1)
        activaciones = []
        x = x + self.pos_emb[:, :seq_len, :]
        for layer in self.layers:
            delta = layer["B"](layer["A"](x))
            # Registramos la magnitud de la contribución por neurona (rank=24)
            # Para simplificar, tomamos el valor absoluto de la salida de layer["A"]
            act = torch.abs(layer["A"](x)).squeeze(0).mean(dim=0) # [rank]
            activaciones.append(act.tolist())
            x = x + delta
        return x, activaciones

def analizar(terreno, chip_name):
    print(f"--- ANALIZANDO FLUJO DINÁMICO: {terreno} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cpu", local_files_only=True)
    embeddings = base_model.get_input_embeddings()
    
    atomo = AtomoGenerico(896, 24, 24)
    atomo.load_state_dict(torch.load(chip_name, map_location="cpu"))
    atomo.eval()
    
    reporte = {"terreno": terreno, "identidades": {}}
    
    with torch.no_grad():
        for id_name, prompt in PARES:
            print(f"  Procesando {id_name} ('{prompt}')...")
            in_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            in_vec = embeddings(in_ids)
            _, activaciones = atomo(in_vec)
            
            # activaciones es [24 capas, 24 neuronas]
            # Calculamos Pos-Vel sobre la magnitud total del flujo
            posiciones = [math.sqrt(sum(n**2 for n in capa)) for capa in activaciones]
            velocidades = [posiciones[i+1] - posiciones[i] for i in range(len(posiciones)-1)]
            
            n = len(velocidades)
            pos = posiciones[:-1]
            media_p, media_v = sum(pos)/n, sum(velocidades)/n
            cov = sum((pos[i] - media_p) * (velocidades[i] - media_v) for i in range(n))
            std_p = math.sqrt(sum((p - media_p) ** 2 for p in pos))
            std_v = math.sqrt(sum((v - media_v) ** 2 for v in velocidades))
            correlacion = cov / (std_p * std_v) if (std_p * std_v) > 0 else 0
            
            reporte["identidades"][id_name] = {
                "correlacion": round(correlacion, 6),
                "posiciones": posiciones,
                "velocidades": velocidades
            }
            
            # Generar Gráfico Individual
            plt.figure(figsize=(8, 5))
            plt.plot(pos, velocidades, 'o-', label=f"Id {id_name}")
            plt.title(f"Flujo {id_name} en {terreno}\nPearson: {round(correlacion, 6)}")
            plt.xlabel("Intensidad de Activación (Pos)")
            plt.ylabel("Variación (Vel)")
            plt.grid(True)
            plt.savefig(f"FLUJO_{id_name}_{terreno}.png")
            plt.close()

    with open(f"REPORTE_FLUJO_{terreno}.json", "w") as f:
        json.dump(reporte, f, indent=4)
    print(f"✅ Análisis dinámico completado para {terreno}.")

if __name__ == "__main__":
    import sys
    if "FRACTAL" in os.getcwd():
        analizar("MANDELBROT", "ATOMO_ABC_MANDELBROT.pt")
    elif "GEOMETRICO" in os.getcwd():
        analizar("GEOMETRICO", "ATOMO_ABC_GEOMETRICO_ARMONICO.pt")
