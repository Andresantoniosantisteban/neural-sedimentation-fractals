import json
import os
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), 'ADN_RAW', 'ADN_TOTAL_IDENTIDADES.json')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'RESULTADOS_PRUEBA_3_1.json')

def obtener_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prueba_3_1_ejecutar():
    """
    PRUEBA 3.1 (ESTADÍSTICA): CENSO DE HIATOS POR CUARTIL INFERIOR
    
    DEFINICIÓN:
    No buscamos el cero absoluto, sino la 'Insignificancia Funcional'. 
    Definimos HIATO como cualquier neurona que pertenezca al primer cuartil (Q1) 
    de impacto absoluto (el 25% de neuronas con menos peso en la identidad).
    
    OBJETIVO:
    Ver si ese 25% de 'silencio' está distribuido uniformemente o si forma 
    estratos (bloques) vacíos que actúan como aislantes semánticos.
    """
    
    print(f"[{obtener_timestamp()}] Iniciando Prueba 3.1: Análisis de Hiatos por Cuartil Inferior (Q1)...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: ADN Total no encontrado en {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resultados_prueba = {
        "timestamp": obtener_timestamp(),
        "hipotesis": "Distribución de Hiatos por Cuartil Inferior (Q1)",
        "metodo": "Filtrado de neuronas en el percentil 25 de impacto absoluto",
        "sujetos_analizados": {}
    }

    for pregunta, contenido in data.items():
        sujeto = contenido['sujeto']
        tokens_data = contenido['analisis_tokens']
        
        for t_info in tokens_data:
            token_str = t_info['token']
            mapa = t_info['mapa_completo']
            
            # 1. Calculamos la intensidad absoluta de toda la población
            intensidades = np.array([abs(n['im']) for n in mapa])
            
            # 2. Calculamos el Umbral del Cuartil Inferior (Q1)
            umbral_q1 = float(np.percentile(intensidades, 25))
            media_impacto = float(np.mean(intensidades))
            
            # 3. Censo de Hiatos (neuronas por debajo del umbral Q1)
            hiatos_ids = [n['i'] for n in mapa if abs(n['im']) <= umbral_q1]
            conteo_hiatos = len(hiatos_ids)
            
            # 4. Mapa de Silencio por Capa
            silencio_por_capa = [0] * 24
            for n in mapa:
                if abs(n['im']) <= umbral_q1:
                    silencio_por_capa[n['c']] += 1
            
            id_key = f"{sujeto}_{token_str}"
            resultados_prueba["sujetos_analizados"][id_key] = {
                "umbral_silencio_q1": umbral_q1,
                "intensidad_media": media_impacto,
                "conteo_hiatos": conteo_hiatos,
                "porcentaje_silencio": (conteo_hiatos / len(mapa)) * 100,
                "mapa_silencio_capas": silencio_por_capa
            }

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(resultados_prueba, f, indent=4)
        
    print(f"[{obtener_timestamp()}] Prueba 3.1 Finalizada.")
    print(f"Informe de Silencio Estadístico generado en {OUTPUT_REPORT}")

if __name__ == "__main__":
    prueba_3_1_ejecutar()

