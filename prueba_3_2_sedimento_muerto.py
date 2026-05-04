import json
import os
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'ADN_TOTAL_IDENTIDADES.json')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'RESULTADOS_PRUEBA_3_2.json')

def obtener_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prueba_3_2_ejecutar():
    """
    PRUEBA 3.2: ANÁLISIS DE SEDIMENTO MUERTO (INTERSECCIÓN DE HIATOS)
    
    DEFINICIÓN CIENTÍFICA:
    El 'Sedimento Muerto' son las neuronas que pertenecen al Cuartil Inferior (Q1) 
    en todos los sujetos analizados. Son zonas de la arquitectura que no participan 
    en la definición de identidades.
    
    OBJETIVO:
    Discernir entre:
    1. Vacío Estructural: Neuronas apagadas para TODOS (Límites del hardware).
    2. Frontera Conceptual: Neuronas apagadas solo para UN sujeto (Límites de la idea).
    """
    
    print(f"[{obtener_timestamp()}] Iniciando Prueba 3.2: Análisis de Sedimento Muerto...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: ADN Total no encontrado.")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    todos_los_hiatos = {}

    for pregunta, contenido in data.items():
        sujeto = contenido['sujeto']
        tokens_data = contenido['analisis_tokens']
        
        for t_info in tokens_data:
            token_str = t_info['token']
            mapa = t_info['mapa_completo']
            
            # Calculamos el umbral Q1 para este token específico
            intensidades = np.array([abs(n['im']) for n in mapa])
            umbral_q1 = np.percentile(intensidades, 25)
            
            # Guardamos el conjunto de IDs de hiatos (Capa, NeuronaID)
            hiatos_set = set([(n['c'], n['i']) for n in mapa if abs(n['im']) <= umbral_q1])
            
            id_key = f"{sujeto}_{token_str}"
            todos_los_hiatos[id_key] = hiatos_set

    # 1. Intersección Universal: Neuronas que son hiato en TODAS las preguntas
    interseccion_universal = set.intersection(*todos_los_hiatos.values())
    
    # 2. Análisis de Exclusividad (Gato vs Sol)
    gato_keys = [k for k in todos_los_hiatos.keys() if 'gato' in k]
    sol_keys = [k for k in todos_los_hiatos.keys() if 'sol' in k]
    
    ejemplo_exclusividad = {}
    if gato_keys and sol_keys:
        k_gato = gato_keys[0]
        k_sol = sol_keys[0]
        silencio_gato = todos_los_hiatos[k_gato]
        silencio_sol = todos_los_hiatos[k_sol]
        
        ejemplo_exclusividad = {
            "sujeto_a": k_gato,
            "sujeto_b": k_sol,
            "hiatos_compartidos": len(silencio_gato & silencio_sol),
            "hiatos_exclusivos_a": len(silencio_gato - silencio_sol),
            "hiatos_exclusivos_b": len(silencio_sol - silencio_gato)
        }

    resultados = {
        "timestamp": obtener_timestamp(),
        "hipotesis": "Diferenciación entre Vacío Estructural y Frontera Conceptual",
        "total_poblacion_modelo": 116736,
        "conteo_sedimento_muerto_universal": len(interseccion_universal),
        "porcentaje_muerto_universal": (len(interseccion_universal) / 116736) * 100,
        "analisis_exclusividad": ejemplo_exclusividad
    }

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=4)
        
    print(f"[{obtener_timestamp()}] Prueba 3.2 Finalizada.")
    print(f"Sedimento Muerto detectado: {resultados['conteo_sedimento_muerto_universal']} neuronas universales.")

if __name__ == "__main__":
    prueba_3_2_ejecutar()
