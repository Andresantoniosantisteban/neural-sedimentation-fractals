import json
import os
from datetime import datetime
from collections import Counter

# --- CONFIGURACIÓN LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'ADN_TOTAL_IDENTIDADES.json')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'RESULTADOS_PRUEBA_2_1.json')

def obtener_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prueba_2_1_ejecutar():
    """
    PRUEBA 2.1: TRAZADO DE CAUCES (HUELLA DE PERSISTENCIA)
    
    DEFINICIÓN CIENTÍFICA:
    Buscamos las neuronas "Inmortales". En un flujo sedimentario, el cauce es la ruta 
    que permanece activa a pesar del movimiento del agua. 
    
    ¿QUÉ ES UNA MARCA DE PERSISTENCIA?:
    Es la huella que dejan los IDs de neuronas que logran mantenerse en el Top de Impacto 
    a través de múltiples capas. Son los eslabones que forman la cadena funcional.
    """
    
    print(f"[{obtener_timestamp()}] Iniciando Prueba 2.1: Trazado de Cauces (Persistencia)...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: No se encuentra el ADN Total en {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resultados_prueba = {
        "timestamp": obtener_timestamp(),
        "hipotesis": "Persistencia de Cauce Neuronal (La Cadena)",
        "metodo": "Censo de Neuronas Inmortales inter-capas (Top 100 por Capa)",
        "sujetos_analizados": {}
    }

    for pregunta, contenido in data.items():
        sujeto = contenido['sujeto']
        tokens_data = contenido['analisis_tokens']
        
        for t_info in tokens_data:
            token_str = t_info['token']
            mapa = t_info['mapa_completo']
            
            # Organización por capas (Clave 'c' en el JSON)
            capas = {}
            for n in mapa:
                l = n['c']
                if l not in capas: capas[l] = []
                capas[l].append(n)
            
            # Extracción de la Élite (Top 100) por cada nivel del río
            todas_las_top_ids = []
            for l in range(24):
                if l in capas:
                    # Ordenamos por impacto estructural ('im')
                    top = sorted(capas[l], key=lambda x: x['im'], reverse=True)[:100]
                    # El ID está en la clave 'i'
                    todas_las_top_ids.extend([n['i'] for n in top])
            
            # CONTEO DE PERSISTENCIA:
            conteo = Counter(todas_las_top_ids)
            
            # Definimos Inmortales como neuronas que aparecen en el Top en >= 3 capas distintas
            inmortales = {str(nid): c for nid, c in conteo.items() if c >= 3}
            # Ordenamos por supervivencia
            inmortales_sorted = dict(sorted(inmortales.items(), key=lambda x: x[1], reverse=True))
            
            id_key = f"{sujeto}_{token_str}"
            resultados_prueba["sujetos_analizados"][id_key] = {
                "poblacion_elite_total": len(conteo),
                "conteo_inmortales": len(inmortales),
                "top_10_inmortales": list(inmortales_sorted.items())[:10]
            }

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(resultados_prueba, f, indent=4)
        
    print(f"[{obtener_timestamp()}] Prueba 2.1 Finalizada.")
    primer_sujeto = list(resultados_prueba["sujetos_analizados"].values())[0]
    print(f"Resultado Ejemplo: {primer_sujeto['conteo_inmortales']} neuronas inmortales detectadas.")

if __name__ == "__main__":
    prueba_2_1_ejecutar()
