import json
import numpy as np
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUE_ES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'ADN_RAW', 'que_es')
OUTPUT_PATH = os.path.join(BASE_DIR, 'acuifero_estructural.json')

def extraer_acuifero():
    archivos = glob.glob(os.path.join(QUE_ES_DIR, "ADN_RAW_*.json"))
    perfiles = []
    
    for f_path in archivos:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        h_tmp = np.zeros(24)
        num_t = len(data['flujo_total'])
        for t_info in data['flujo_total']:
            for capa_info in t_info['capas']:
                capa = capa_info['capa']
                impacto = sum([abs(n['im']) for n in capa_info['flujo']])
                h_tmp[capa] += impacto
        
        perfiles.append(h_tmp / num_t)
    
    topografia_media = np.mean(perfiles, axis=0)
    topografia_norm = topografia_media / (np.mean(topografia_media) + 1e-9)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "topografia_media": topografia_media.tolist(),
            "topografia_norm": topografia_norm.tolist()
        }, f, indent=4)
    
    print(f"Acuífero Estructural guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    extraer_acuifero()
