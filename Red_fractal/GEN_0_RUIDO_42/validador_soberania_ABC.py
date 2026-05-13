# validador_soberania_ABC.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

# CONFIGURACIÓN
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CHIP_PATH = "c:/Users/andre/Desktop/Neural_Identity_Forge/EN_DESARROLLO/FABRICA_ATOMOS_CRISTALINOS_17N/ATOMO_ABC_24x24_coherente.pt"
PARES = [
    ("Hola", "¿Cómo estás?"),
    ("punto", "Hasta luego"),
    ("saber", "No es creer")
]

class Atomo24x24(nn.Module):
    def __init__(self, dim, rank, layers, max_seq=5):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "A": nn.Linear(dim, rank, bias=False),
                "B": nn.Linear(rank, dim, bias=False)
            }) for _ in range(layers)
        ])
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        for layer in self.layers:
            x = x + layer["B"](layer["A"](x))
        return x

def validar():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cpu", local_files_only=True)
    embeddings = base_model.get_input_embeddings()
    lm_head = base_model.get_output_embeddings()

    atomo = Atomo24x24(896, 24, 24, max_seq=5)
    atomo.load_state_dict(torch.load(CHIP_PATH))
    atomo.eval()

    resultados = []
    soberania_global = True

    with torch.no_grad():
        for prompt, target in PARES:
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            slen = len(target_ids)
            
            input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            input_vec = embeddings(input_ids)[:, :1, :].expand(-1, slen, -1)
            res_vec = atomo(input_vec)
            logits = lm_head(res_vec)
            
            probs = torch.softmax(logits, dim=-1)
            max_probs, pred_ids = torch.max(probs, dim=-1)
            resultado = tokenizer.decode(pred_ids[0])
            confianza_media = max_probs[0].mean().item()
            
            exito = (resultado.strip() == target.strip())
            if not exito: soberania_global = False
            
            resultados.append({
                "prompt": prompt,
                "objetivo": target,
                "obtenido": resultado.strip(),
                "confianza": f"{confianza_media:.6f}",
                "estado": "SOBERANO" if exito else "FALLIDO"
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"c:/Users/andre/Desktop/Neural_Identity_Forge/EN_DESARROLLO/FABRICA_ATOMOS_CRISTALINOS_17N/{timestamp}_VAL_SOBERANIA_ABC_24x24.json"

    reporte = {
        "fecha_validacion": timestamp,
        "chip_archivo": "ATOMO_ABC_24x24_coherente.pt",
        "soberania_total": soberania_global,
        "detalles": resultados
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=4, ensure_ascii=False)

    print(f"\n✅ VALIDACIÓN COMPLETADA")
    print(f"Reporte generado: {filename}")
    return reporte

if __name__ == "__main__":
    validar()
