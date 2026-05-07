import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Qwen2Config, Qwen2ForCausalLM
import json
import os
from datetime import datetime

# =================================================================
# Autor: Andrés Antonio Santisteban Lino
# Operación: Cristalización con Log Detallado de Causalidad
# Tarea: n -> (n % 3) con flujo corregido (Sutura Causal)
# Objetivo: Obtener el archivo .pt Y el mapa .json de movimientos.
# =================================================================

def obtener_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")

def realizar_microscopia():
    timestamp = obtener_timestamp()
    folder_path = os.path.join("EN_DESARROLLO", "MICROSCOPIA_NEURONAL")
    heartbeat_path = os.path.join(folder_path, f"{timestamp}_HEARTBEAT_CAUSAL.txt")
    
    # Red Fresca
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False
    )
    
    model = Qwen2ForCausalLM(config)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.15) # Ajuste fino de LR
    
    print(f"Lanzando Operación Cristalización con Log: {timestamp}")
    
    max_epocas = 150
    umbral_cristal = 0.05
    numeros = [i for i in range(60)]
    
    log_causalidad = []
    log_pulso = []
    umbral_movimiento = 1e-15

    for epoca in range(max_epocas):
        perdida_epoca = 0
        indices = torch.randperm(len(numeros)).tolist()
        
        for idx in indices:
            id_paso = f"E{epoca:03d}_N{idx:03d}"
            optimizer.zero_grad()
            
            # Captura PRE
            sed_pre = model.get_input_embeddings().weight[idx].clone().detach()
            
            # Forward Causal
            input_ids = torch.tensor([[idx, 0]], dtype=torch.long)
            labels = torch.tensor([[-100, (idx % 3)]], dtype=torch.long)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # Captura Gradiente (Fuerza)
            grad_fuerza = model.get_input_embeddings().weight.grad[idx].clone().detach()
            
            optimizer.step()
            
            # Captura POST (Movimiento)
            sed_post = model.get_input_embeddings().weight[idx].clone().detach()
            delta = sed_post - sed_pre
            distancia = torch.norm(delta, p=2).item()
            
            if distancia > umbral_movimiento:
                log_causalidad.append({
                    "id": id_paso,
                    "token": idx,
                    "grupo": idx % 3,
                    "delta": delta.tolist(),
                    "fuerza": grad_fuerza.tolist(),
                    "dist": distancia
                })
            
            perdida_epoca += loss.item()
            
        media_loss = perdida_epoca / len(numeros)
        
        # Guardado del Pulso (Promedio por época para no saturar)
        log_pulso.append({
            "epoca": epoca,
            "loss": media_loss
        })
        
        # Latido
        with open(heartbeat_path, "w") as hb:
            hb.write(f"SUTURANDO - Epoca {epoca} - Loss: {media_loss:.6f} - Movs: {len(log_causalidad)}")
        
        if epoca % 10 == 0 or media_loss < umbral_cristal:
            print(f"[Epoca {epoca:03d}] Loss: {media_loss:.4f}")

        if media_loss < umbral_cristal:
            print(f"\n¡CRISTALIZACIÓN! Loss: {media_loss:.6f}")
            break

    # --- GUARDADO MAESTRO ---
    torch.save(model.state_dict(), os.path.join(folder_path, f"{timestamp}_CRISTALIZADO_FINAL.pt"))
    
    with open(os.path.join(folder_path, f"{timestamp}_MAPA_CAUSALIDAD.json"), "w") as f:
        json.dump(log_causalidad, f, indent=4)
        
    with open(os.path.join(folder_path, f"{timestamp}_PULSO_ENTRENAMIENTO.json"), "w") as f:
        json.dump(log_pulso, f, indent=4)

    print(f"Operación finalizada. Archivos generados con éxito.")

if __name__ == "__main__":
    realizar_microscopia()
