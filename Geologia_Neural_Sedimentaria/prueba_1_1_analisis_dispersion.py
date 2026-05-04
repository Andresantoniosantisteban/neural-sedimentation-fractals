import json
import numpy as np
import os
from scipy import stats
from datetime import datetime

# --- CONFIGURACIÓN DE RUTAS LOCALES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), 'ADN_RAW', 'ADN_TOTAL_IDENTIDADES.json')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'RESULTADOS_PRUEBA_1_1.json')

def obtener_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prueba_1_1_ejecutar():
    """
    PRUEBA 1.1: ANÁLISIS DE DISPERSIÓN FRACTAL (LEY DE ZIPF)
    
    DEFINICIÓN CIENTÍFICA:
    Esta prueba busca validar si la distribución del impacto neuronal en el modelo sigue una 
    Ley de Potencia (Power Law). En la naturaleza, los sistemas que se auto-organizan mediante 
    procesos de agregación (como el DLA - Diffusion-Limited Aggregation) muestran un exponente 
    Alpha cercano a 1.0. 
    
    ¿POR QUÉ ES RIGUROSO ESTE TEST?:
    1. Escala Log-Log: Al pasar los datos a escala logarítmica, cualquier relación de potencia 
       se convierte en una línea recta. La pendiente de esa línea es el "ADN fractal" del concepto.
    2. Universalidad: Se analiza el 100% del espectro neuronal, no solo el Top-100, eliminando 
       el sesgo de selección.
    3. Bondad de Ajuste (R2): Mide qué tan "ruidoso" es el fractal. Un R2 > 0.8 indica que el 
       orden es determinista y no fruto del azar.
    """
    
    print(f"[{obtener_timestamp()}] Iniciando Prueba 1.1: Análisis de Dispersión Fractal Total...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: No se encuentra el ADN Total en {INPUT_PATH}")
        return

    # Carga de la base de datos maestra (873 MB aprox.)
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resultados_prueba = {
        "timestamp": obtener_timestamp(),
        "metodo": "Regresión Lineal sobre Distribución de Rango-Frecuencia (Zipf)",
        "hipotesis": "DLA / Floculación Fractal",
        "fundamento": "Si Alpha ~ 1.0, la identidad es un fractal auto-organizado por flujo acumulativo.",
        "sujetos_analizados": {}
    }

    alphas_encontrados = []

    for pregunta, contenido in data.items():
        sujeto = contenido['sujeto']
        tokens_data = contenido['analisis_tokens']
        
        for t_info in tokens_data:
            token_str = t_info['token']
            # Extraemos el impacto estructural (im) de cada neurona activa (>0)
            # Esto representa el "sedimento" depositado en esa coordenada neuronal.
            impactos = [n['im'] for n in t_info['mapa_completo'] if n['im'] > 0]
            
            # Umbral de población mínima para que la estadística sea válida
            if len(impactos) < 100: continue
            
            # ORDENAMIENTO DE RANGO:
            # Las neuronas se ordenan de mayor a menor impacto para crear la curva de Zipf.
            impactos_sorted = sorted(impactos, reverse=True)
            
            # TRANSFORMACIÓN LOGARÍTMICA:
            # log(Rango) vs log(Impacto)
            log_rangos = np.log10(np.arange(1, len(impactos_sorted) + 1))
            log_impactos = np.log10(impactos_sorted)
            
            # REGRESIÓN LINEAL:
            # Buscamos la pendiente (slope) que define la tasa de caída de energía del fractal.
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_rangos, log_impactos)
            
            # ALPHA: El valor absoluto de la pendiente. 
            # Alpha = 1.0 es el equilibrio Zipfiano perfecto.
            alpha = abs(slope)
            r2 = r_value**2
            
            alphas_encontrados.append(alpha)
            
            id_key = f"{sujeto}_{token_str}"
            resultados_prueba["sujetos_analizados"][id_key] = {
                "alpha": alpha, # Dimensión fractal de la distribución
                "r2": r2,       # Precisión del modelo fractal
                "poblacion_viva": len(impactos), # Número de neuronas participando
                "estabilidad_zipf": "ALTA" if r2 > 0.8 else "MEDIA" if r2 > 0.6 else "BAJA"
            }

    # ANÁLISIS GLOBAL:
    # Promediamos todos los sujetos para ver si el comportamiento es una constante del modelo.
    resultados_prueba["resumen_cientifico"] = {
        "alpha_medio": float(np.mean(alphas_encontrados)),
        "desviacion_estandar": float(np.std(alphas_encontrados)),
        "veredicto_fractal": "CONFIRMADO" if 0.8 <= np.mean(alphas_encontrados) <= 1.2 else "DISCUTIBLE"
    }

    # Persistencia en JSON para documentación forense
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(resultados_prueba, f, indent=4)
        
    print(f"[{obtener_timestamp()}] Prueba 1.1 Finalizada.")
    print(f"Veredicto: {resultados_prueba['resumen_cientifico']['veredicto_fractal']}")
    print(f"Alpha Medio: {resultados_prueba['resumen_cientifico']['alpha_medio']:.4f}")

if __name__ == "__main__":
    # Ejecución determinista del test de validación
    prueba_1_1_ejecutar()

