# forja_triade_ABC_24x24_coherente.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random
import numpy as np

# --- PROTOCOLO 42 ---
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# --- CONFIGURACIÓN ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PARES = [
    ("Hola", "¿Cómo estás?"),     # A
    ("punto", "Hasta luego"),    # B
    ("saber", "No es creer")     # C
]
RANK = 24
LAYERS = 24
DIM = 896
EPOCHS = 20001
LEARNING_RATE = 0.00001
LOG_FILE = "c:/Users/andre/Desktop/Neural_Identity_Forge/EN_DESARROLLO/FABRICA_ATOMOS_CRISTALINOS_17N/log_ABC_24x24_coherente.txt"
CHIP_PATH = "c:/Users/andre/Desktop/Neural_Identity_Forge/EN_DESARROLLO/FABRICA_ATOMOS_CRISTALINOS_17N/ATOMO_ABC_24x24_coherente.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log_print(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

log_print(f"🔱 INICIANDO FORJA DE LA TRÍADE ABC (24x24) - DESDE CERO - LEY DE COHERENCIA - PROTOCOLO 42")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map=DEVICE, local_files_only=True)
embeddings = base_model.get_input_embeddings()

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

# Preparar datos
max_target_len = 5 # Suficiente para las 3 identidades
training_data = []
for prompt, resp in PARES:
    in_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    out_ids = tokenizer.encode(resp, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    target_vecs = embeddings(out_ids).detach()
    training_data.append((in_ids, target_vecs, out_ids.size(1)))

atomo = Atomo24x24(DIM, RANK, LAYERS, max_seq=max_target_len).to(DEVICE)
optimizer = optim.Adam(atomo.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, threshold=0.01, verbose=True)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    indiv_losses = []
    
    for in_ids, target_vecs, seq_len in training_data:
        input_vec = embeddings(in_ids)[:, :1, :].expand(-1, seq_len, -1)
        output_vectors = atomo(input_vec)
        loss = criterion(output_vectors, target_vecs)
        indiv_losses.append(loss)
    
    # LEY DE COHERENCIA DE ANDRÉS (A, B, C)
    losses_tensor = torch.stack(indiv_losses)
    mean_loss = losses_tensor.mean()
    std_loss = losses_tensor.std()
    
    total_loss_val = mean_loss + std_loss 
    
    total_loss_val.backward()
    optimizer.step()
    
    current_loss = mean_loss.item()
    scheduler.step(current_loss)

    if epoch % 500 == 0:
        curr_lr = optimizer.param_groups[0]['lr']
        log_print(f"Epoch {epoch} | Loss: {current_loss:.12f} | Std: {std_loss.item():.12f} | LR: {curr_lr}")
        # AUTO-GUARDADO DE SEGURIDAD
        torch.save(atomo.state_dict(), CHIP_PATH)

torch.save(atomo.state_dict(), CHIP_PATH)
log_print("✅ Forja de la Tríade ABC 24x24 Completada.")
