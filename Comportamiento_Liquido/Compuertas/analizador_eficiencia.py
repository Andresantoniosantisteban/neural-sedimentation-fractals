import json
import os
from datetime import datetime

def analizar_eficiencia(source_file):
    if not os.path.exists(source_file):
        print(f"Error: No se encuentra el archivo {source_file}")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    resultados_comparativa = []

    for item in data['auditoria']:
        pregunta = item['pregunta']
        barrido = item['barrido']
        
        # 1. Obtener estado Basal (Presión 0.0)
        basal = next((b for b in barrido if b['presion'] == 0.0), None)
        if not basal: continue
        
        potencia_basal = sum(capa['potencia_avg'] for capa in basal['metricas_capa'].values())
        
        # 2. Obtener el mejor estado en zona de VACÍO (Presión negativa)
        vacios = [b for b in barrido if b['presion'] < 0.0]
        if not vacios: continue
        
        mejor_vacio = max(vacios, key=lambda x: (x['score_gpt'], -sum(capa['potencia_avg'] for capa in x['metricas_capa'].values())))
        potencia_vacio = sum(capa['potencia_avg'] for capa in mejor_vacio['metricas_capa'].values())
        
        ahorro_energia = ((potencia_vacio - potencia_basal) / potencia_basal) * 100
        incremento_score = mejor_vacio['score_gpt'] - basal['score_gpt']

        resultados_comparativa.append({
            "pregunta": pregunta,
            "basal": {
                "score": basal['score_gpt'],
                "potencia": round(potencia_basal, 4),
                "respuesta": basal['respuesta']
            },
            "vacio_optimo": {
                "presion": mejor_vacio['presion'],
                "score": mejor_vacio['score_gpt'],
                "potencia": round(potencia_vacio, 4),
                "respuesta": mejor_vacio['respuesta']
            },
            "ahorro_pct": round(ahorro_energia, 2),
            "mejora_score": incremento_score
        })

    return resultados_comparativa

if __name__ == "__main__":
    import glob
    COMPUERTAS_DIR = r"c:\Users\andre\Desktop\Neural_Identity_Forge\Comportamiento_Liquido\Compuertas"
    # Buscar el reporte FINAL más reciente
    files = glob.glob(os.path.join(COMPUERTAS_DIR, "*_AUDITORIA_COMPUERTAS_FINAL.json"))
    if not files:
        print("No se encontraron reportes finales.")
    else:
        latest_file = max(files, key=os.path.getctime)
        print(f"Analizando: {os.path.basename(latest_file)}")
        reporte = analizar_eficiencia(latest_file)
        
        # Guardar el JSON de comparativa
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = os.path.join(COMPUERTAS_DIR, f"{timestamp}_comparativa_EFICIENCIA_FINAL.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"metadatos": {"autor": "Andrés Antonio Santisteban Lino", "timestamp": timestamp}, "comparativa": reporte}, f, indent=4, ensure_ascii=False)
            
        # Resumen en consola
        print("\n--- RESUMEN DE SOBERANÍA ENERGÉTICA (VACÍO vs BASAL) ---")
        print(f"{'PREGUNTA':<40} | {'DELTA SCORE':<12} | {'AHORRO POTENCIA':<15}")
        print("-" * 75)
        for res in reporte:
            p = res['pregunta'][:37] + "..."
            ds = res['mejora_score']
            ap = res['ahorro_pct']
            print(f"{p:<40} | {ds:<12} | {ap:>14}%")
        print(f"\nReporte comparativo generado: {os.path.basename(output_path)}")
