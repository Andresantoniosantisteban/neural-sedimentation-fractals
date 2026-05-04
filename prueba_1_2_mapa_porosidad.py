import json
import numpy as np
import os
from datetime import datetime

# --- CONFIGURACIÓN DE RUTAS LOCALES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'ADN_TOTAL_IDENTIDADES.json')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'RESULTADOS_PRUEBA_1_2.json')

def obtener_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prueba_1_2_ejecutar():
    """
    PRUEBA 1.2: MAPA DE POROSIDAD (ÍNDICE DE CONCENTRACIÓN DE IMPACTO)
    
    DEFINICIÓN CIENTÍFICA:
    En geología, la porosidad mide el espacio vacío en una roca. En nuestra "Geología Neuronal", 
    la porosidad mide cuánta "superficie" de la identidad está vacía de impacto real. 
    
    ¿POR QUÉ BUSCAMOS ESTO?:
    Si la identidad es un fractal DLA (una pelusa o flóculo), deberíamos observar que un 
    número mínimo de neuronas (el núcleo del flóculo) concentra casi todo el impacto semántico, 
    dejando el resto de la estructura como un soporte poroso de baja energía.
    
    MÉTRICA DE VALIDACIÓN:
    Calcularemos cuántas neuronas se necesitan para alcanzar el 50%, 80% y 90% del impacto total.
    - Si el 90% del impacto reside en menos del 20% de las neuronas activas, confirmamos que la 
      identidad es un objeto poroso y no una masa sólida.
    """
    
    print(f"[{obtener_timestamp()}] Iniciando Prueba 1.2: Mapa de Porosidad...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: No se encuentra el ADN Total en {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resultados_prueba = {
        "timestamp": obtener_timestamp(),
        "hipotesis": "Porosidad del Flóculo Fractal",
        "metodo": "Análisis de Acumulación de Energía (Pareto Neuronal)",
        "descripcion": "Medición de la densidad de impacto sobre la población neuronal activa.",
        "sujetos_analizados": {}
    }

    porosidades_globales = []

    for pregunta, contenido in data.items():
        sujeto = contenido['sujeto']
        tokens_data = contenido['analisis_tokens']
        
        for t_info in tokens_data:
            token_str = t_info['token']
            # Extraemos y ordenamos impactos de mayor a menor
            impactos = sorted([n['im'] for n in t_info['mapa_completo'] if n['im'] > 0], reverse=True)
            
            if not impactos: continue
            
            total_impacto = sum(impactos)
            acumulado = 0
            n_50, n_80, n_90 = 0, 0, 0
            
            # Buscamos los puntos de saturación de energía
            for i, val in enumerate(impactos):
                acumulado += val
                porcentaje_acumulado = (acumulado / total_impacto) * 100
                
                # Identificamos cuántas neuronas se requieren para cada hito de "realidad"
                if n_50 == 0 and porcentaje_acumulado >= 50: n_50 = i + 1
                if n_80 == 0 and porcentaje_acumulado >= 80: n_80 = i + 1
                if n_90 == 0 and porcentaje_acumulado >= 90: n_90 = i + 1
            
            # ÍNDICE DE POROSIDAD:
            # Representa el porcentaje de la población activa que NO es crítica para el 90% del impacto.
            # Una porosidad del 90% significa que solo el 10% de las neuronas vivas son el "corazón" del concepto.
            indice_porosidad = (1 - (n_90 / len(impactos))) * 100
            porosidades_globales.append(indice_porosidad)
            
            id_key = f"{sujeto}_{token_str}"
            resultados_prueba["sujetos_analizados"][id_key] = {
                "poblacion_viva": len(impactos),
                "neuronas_para_50_pct": n_50,
                "neuronas_para_80_pct": n_80,
                "neuronas_para_90_pct": n_90,
                "porosidad_semantica_pct": indice_porosidad
            }

    # ESTADÍSTICAS GLOBALES
    resultados_prueba["resumen_cientifico"] = {
        "porosidad_media": float(np.mean(porosidades_globales)),
        "eficiencia_estructural": "EXTREMA" if np.mean(porosidades_globales) > 90 else "ALTA",
        "veredicto": "POROSIDAD FRACTAL CONFIRMADA" if np.mean(porosidades_globales) > 75 else "ESTRUCTURA SEMI-SÓLIDA"
    }

    # Guardado de la evidencia en JSON
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(resultados_prueba, f, indent=4)
        
    print(f"[{obtener_timestamp()}] Prueba 1.2 Finalizada.")
    print(f"Porosidad Media: {resultados_prueba['resumen_cientifico']['porosidad_media']:.2f}%")
    print(f"Veredicto: {resultados_prueba['resumen_cientifico']['veredicto']}")

if __name__ == "__main__":
    prueba_1_2_ejecutar()
