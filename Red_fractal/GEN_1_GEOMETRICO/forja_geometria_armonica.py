# forja_geometria_armonica.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import math

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
EPOCHS = 40001
LEARNING_RATE = 0.0001
FOLDER = "c:/Users/andre/Desktop/Neural_Identity_Forge/EN_DESARROLLO/FABRICA_ATOMOS_CRISTALINOS_17N/GEOMETRICO"
LOG_FILE = f"{FOLDER}/log_ABC_geometria.txt"
CHIP_PATH = f"{FOLDER}/ATOMO_ABC_GEOMETRICO_ARMONICO.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log_print(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

log_print(f"🔱 INICIANDO FORJA - GEOMETRÍA ARMÓNICA (SIN AZAR) - PORTADORA SINUSOIDAL")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map=DEVICE, local_files_only=True)
embeddings = base_model.get_input_embeddings()

class AtomoGeometriaArmonica(nn.Module):
    def __init__(self, dim, rank, layers, max_seq=5):
        super().__init__()
        
        # INICIALIZACIÓN GEOMÉTRICA (Seno/Coseno)
        # Position Embedding Armónico
        pos = torch.arange(max_seq).unsqueeze(1)
        d = torch.arange(dim).unsqueeze(0)
        self.pos_emb = nn.Parameter(torch.sin(pos * d * 0.1).unsqueeze(0) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "A": nn.Linear(dim, rank, bias=False),
                "B": nn.Linear(rank, dim, bias=False)
            }) for _ in range(layers)
        ])
        
        # Inicializar Pesos con Canales Armónicos
        for l_idx, layer in enumerate(self.layers):
            # A: [rank, dim]
            # B: [dim, rank]
            v_rank = torch.sin(torch.arange(rank).float() * (l_idx + 1) * 0.5).unsqueeze(1)
            v_dim = torch.cos(torch.arange(dim).float() * (l_idx + 1) * 0.5).unsqueeze(0)
            
            # Crear malla de interferencia geométrica
            grid = torch.mm(v_rank, v_dim) * 0.02
            
            layer["A"].weight.data.copy_(grid)
            layer["B"].weight.data.copy_(grid.t())
            
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        for layer in self.layers:
            x = x + layer["B"](layer["A"](x))
        return x

# Preparar datos
max_target_len = 5
training_data = []
for prompt, resp in PARES:
    in_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    out_ids = tokenizer.encode(resp, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    target_vecs = embeddings(out_ids).detach()
    training_data.append((in_ids, target_vecs, out_ids.size(1)))

atomo = AtomoGeometriaArmonica(DIM, RANK, LAYERS, max_seq=max_target_len).to(DEVICE)
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
    
    losses_tensor = torch.stack(indiv_losses)
    mean_loss = losses_tensor.mean()
    std_loss = losses_tensor.std() if len(indiv_losses) > 1 else torch.tensor(0.0).to(DEVICE)
    
    total_loss_val = mean_loss + std_loss 
    
    total_loss_val.backward()
    optimizer.step()
    
    current_loss = mean_loss.item()
    scheduler.step(current_loss)
    
    if epoch % 500 == 0:
        curr_lr = optimizer.param_groups[0]['lr']
        log_print(f"Epoch {epoch} | Loss: {current_loss:.12f} | Std: {std_loss.item():.12f} | LR: {curr_lr}")
        torch.save(atomo.state_dict(), CHIP_PATH)

torch.save(atomo.state_dict(), CHIP_PATH)
log_print("✅ Forja GEOMÉTRICA ARMÓNICA Completada.")
