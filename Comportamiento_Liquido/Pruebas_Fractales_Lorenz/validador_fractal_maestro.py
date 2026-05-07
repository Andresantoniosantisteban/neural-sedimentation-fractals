import torch
import json
import os
import numpy as np
from datetime import datetime

# =================================================================
# Autor: Andrés Antonio Santisteban Lino
# Proyecto: Validador Fractal Maestro v6 (Protocolo de Archivo)
# Objetivo: Validar geometrías y grabar obligatoriamente en JSON
#           con marca de tiempo (Timestamp).
# =================================================================

def obtener_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")

def analizar_fractales():
    timestamp = obtener_timestamp()
    folder_path = "EN_DESARROLLO/MICROSCOPIA_NEURONAL"
    archivo_pt = os.path.join(folder_path, "20260506_1155_CRISTALIZADO_FINAL.pt")
    mapa_json_input = os.path.join(folder_path, "20260506_1155_MAPA_CAUSALIDAD.json")
    
    # Nombre de archivo según protocolo: YYYYMMDD_HHMM_nombre.json
    archivo_output = os.path.join(folder_path, f"{timestamp}_sentencia_fractal_maestra.json")
    
    if not os.path.exists(archivo_pt):
        print(json.dumps({"error": f"No se encuentra {archivo_pt}"}))
        return

    # Carga de Datos
    state_dict = torch.load(archivo_pt, weights_only=True)
    embeddings = state_dict['model.embed_tokens.weight'][:60].detach().numpy()
    with open(mapa_json_input, 'r') as f:
        mapa_causalidad = json.load(f)

    # --- MÉTRICAS ---
    centroids = [embeddings[i::3].mean(axis=0) for i in range(3)]
    
    # Cantor
    dist_01 = np.linalg.norm(centroids[0] - centroids[1])
    dist_12 = np.linalg.norm(centroids[1] - centroids[2])
    dist_02 = np.linalg.norm(centroids[0] - centroids[2])
    max_span = np.max([dist_01, dist_12, dist_02])
    ratio_cantor = (dist_01 / max_span) if max_span > 0 else 0
    es_cantor = abs(ratio_cantor - 0.33) < 0.05

    # Sierpinski
    m9 = embeddings[0::9]
    m3 = embeddings[0::3]
    avg_dist_m9 = np.mean([np.linalg.norm(m9[i]-m9[j]) for i in range(len(m9)) for j in range(i+1, len(m9))])
    avg_dist_m3 = np.mean([np.linalg.norm(m3[i]-m3[j]) for i in range(len(m3)) for j in range(i+1, len(m3))])
    ratio_sierp = avg_dist_m9 / avg_dist_m3 if avg_dist_m3 > 0 else 0
    es_sierp = ratio_sierp < 0.75

    # Lorenz
    def cos_sim(v1, v2): return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    sim_lorenz = cos_sim(centroids[0], centroids[2])
    es_lorenz = sim_lorenz > 0.50

    # Julia
    high_impact = [np.linalg.norm(d['delta']) for d in mapa_causalidad if d['dist'] > 0.39]
    idx_julia = np.std(high_impact) / np.mean(high_impact) if len(high_impact)>0 else 0
    es_julia = idx_julia > 1.5

    # Veredicto
    if es_lorenz and es_julia:
        sentencia = "GEOMETRÍA DE LORENZ (Con Frontera de Julia)"
    elif es_cantor:
        sentencia = "GEOMETRÍA TERNARIA DE CANTOR"
    elif es_julia:
        sentencia = "GEOMETRÍA CAÓTICA DE JULIA"
    else:
        sentencia = "GEOMETRÍA NO IDENTIFICADA"

    # Estructura del Resultado
    resultado_final = {
        "metadatos": {
            "autor": "Andrés Antonio Santisteban Lino",
            "timestamp": timestamp,
            "archivo_fuente": archivo_pt
        },
        "sentencia_final": sentencia,
        "validaciones": {
            "cantor": {"status": "CONFIRMADO" if es_cantor else "RECHAZADO", "valor": float(ratio_cantor)},
            "sierpinski": {"status": "CONFIRMADO" if es_sierp else "RECHAZADO", "valor": float(ratio_sierp)},
            "lorenz": {"status": "CONFIRMADO" if es_lorenz else "RECHAZADO", "valor": float(sim_lorenz)},
            "julia": {"status": "CONFIRMADO" if es_julia else "RECHAZADO", "valor": float(idx_julia)}
        },
        "centroides_naciones": [c.tolist() for c in centroids]
    }

    # 1. GUARDADO FÍSICO (OBLIGATORIO)
    with open(archivo_output, "w", encoding='utf-8') as f:
        json.dump(resultado_final, f, indent=4, ensure_ascii=False)

    # 2. SALIDA POR CONSOLA
    print(json.dumps(resultado_final, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    analizar_fractales()
