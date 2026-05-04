import json
import numpy as np
import os
from datetime import datetime

# --- CONFIGURACIÓN LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), 'ADN_RAW', 'ADN_TOTAL_IDENTIDADES.json')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'RESULTADOS_PRUEBA_4_1.json')

def obtener_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prueba_4_1_ejecutar():
    """
    PRUEBA 4.1: DETECCIÓN DEL PUNTO DE ROCÍO (CRISTALIZACIÓN)
    
    DEFINICIÓN CIENTÍFICA:
    El punto de rocío es el umbral térmico donde el gas se condensa. En nuestra 
    investigación, es la capa donde la dispersión fractal colapsa y la energía 
    se concentra en un núcleo crítico.
    
    METODOLOGÍA:
    1. Medimos el Coeficiente de Gini (desigualdad) del impacto por capa.
    2. La capa con el Gini más alto es la capa de mayor 'Cristalización' 
       (donde unas pocas neuronas mandan sobre todo el resto).
    3. Cruzamos este dato con la presencia de 'Inmortales' de la Fase 2.1.
    """
    
    print(f"[{obtener_timestamp()}] Iniciando Prueba 4.1: Detección del Punto de Rocío...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: ADN Total no encontrado.")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resultados_prueba = {
        "timestamp": obtener_timestamp(),
        "hipotesis": "Punto de Rocío Semántico (Colapso de Entropía)",
        "metodo": "Cálculo de Coeficiente de Gini de Impacto por Capa",
        "sujetos_analizados": {}
    }

    for pregunta, contenido in data.items():
        sujeto = contenido['sujeto']
        tokens_data = contenido['analisis_tokens']
        
        for t_info in tokens_data:
            token_str = t_info['token']
            mapa = t_info['mapa_completo']
            
            # Organizamos impactos por capa
            impactos_capas = [[] for _ in range(24)]
            for n in mapa:
                impactos_capas[n['c']].append(abs(n['im']))
            
            ginis = []
            for i_capa in impactos_capas:
                if not i_capa: 
                    ginis.append(0.0)
                    continue
                # Cálculo de Gini simplificado: (Suma de diferencias / n^2 * media)
                sorted_i = np.sort(i_capa)
                n = len(i_capa)
                index = np.arange(1, n + 1)
                gini = ((np.sum((2 * index - n - 1) * sorted_i)) / (n * np.sum(sorted_i)))
                ginis.append(float(gini))
            
            punto_rocio = int(np.argmax(ginis))
            
            id_key = f"{sujeto}_{token_str}"
            resultados_prueba["sujetos_analizados"][id_key] = {
                "perfil_ginis_por_capa": ginis,
                "capa_de_cristalizacion": punto_rocio,
                "intensidad_en_rocio": float(np.max(ginis))
            }

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(resultados_prueba, f, indent=4)
        
    print(f"[{obtener_timestamp()}] Prueba 4.1 Finalizada.")
    cristal_medio = np.mean([s['capa_de_cristalizacion'] for s in resultados_prueba['sujetos_analizados'].values()])
    print(f"Punto de Rocío detectado en torno a la Capa {int(cristal_medio)}.")

if __name__ == "__main__":
    prueba_4_1_ejecutar()

