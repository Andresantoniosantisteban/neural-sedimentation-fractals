import json
import numpy as np
import os
from datetime import datetime

# --- CONFIGURACIÓN LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), 'ADN_RAW', 'ADN_TOTAL_IDENTIDADES.json')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'RESULTADOS_PRUEBA_2_2.json')

def obtener_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prueba_2_2_ejecutar():
    """
    PRUEBA 2.2: MEDICIÓN DE LA PENDIENTE (FLUJO DE ENERGÍA)
    
    DEFINICIÓN CIENTÍFICA:
    En geología, la pendiente determina la capacidad erosiva y de transporte del agua. 
    Aquí mediremos la "Presión Semántica" por capa para ver cómo el modelo empuja la idea.
    
    ¿QUÉ BUSCAMOS?:
    - Presas (Embalses): Capas con picos de impacto total. Indican zonas de alta computación.
    - Cascadas: Saltos donde el impacto cae bruscamente. Indican zonas de filtrado o simplificación.
    """
    
    print(f"[{obtener_timestamp()}] Iniciando Prueba 2.2: Medición de Pendiente...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: ADN Total no encontrado en {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resultados_prueba = {
        "timestamp": obtener_timestamp(),
        "hipotesis": "Perfil de Presión Semántica y Gradientes de Flujo",
        "metodo": "Análisis de impacto acumulado por nivel (Layer-wise Cumulative Impact)",
        "sujetos_analizados": {}
    }

    for pregunta, contenido in data.items():
        sujeto = contenido['sujeto']
        tokens_data = contenido['analisis_tokens']
        
        for t_info in tokens_data:
            token_str = t_info['token']
            mapa = t_info['mapa_completo']
            
            # Sumamos el impacto total de TODAS las neuronas de cada capa
            impacto_por_capa = [0.0] * 24
            for n in mapa:
                impacto_por_capa[n['c']] += n['im']
            
            # Calculamos los gradientes (la diferencia de presión entre capas consecutivas)
            gradientes = [impacto_por_capa[i+1] - impacto_por_capa[i] for i in range(23)]
            
            # Identificamos el "Embalse Maestro" (Capa con más energía)
            capa_max_presion = int(np.argmax(impacto_por_capa))
            # Identificamos la "Cascada Crítica" (Mayor caída de impacto)
            capa_max_caida = int(np.argmin(gradientes))
            
            id_key = f"{sujeto}_{token_str}"
            resultados_prueba["sujetos_analizados"][id_key] = {
                "perfil_energia": [float(x) for x in impacto_por_capa],
                "gradiente_intercapa": [float(x) for x in gradientes],
                "punto_de_max_presion": capa_max_presion,
                "punto_de_cascada_critica": capa_max_caida,
                "presion_media": float(np.mean(impacto_por_capa))
            }

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(resultados_prueba, f, indent=4)
        
    print(f"[{obtener_timestamp()}] Prueba 2.2 Finalizada.")
    print(f"Informe generado en {OUTPUT_REPORT}")

if __name__ == "__main__":
    prueba_2_2_ejecutar()

