# forjador_fractal_maestro.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

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
FOLDER = "c:/Users/andre/Desktop/Neural_Identity_Forge/EN_DESARROLLO/FABRICA_ATOMOS_CRISTALINOS_17N/FRACTAL"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GeneradorFractal:
    @staticmethod
    def mandelbrot(h, w, max_iter=20):
        # Genera una matriz basada en el conjunto de Mandelbrot
        y, x = np.ogrid[-1.5:1.5:h*1j, -2:1:w*1j]
        c = x + y*1j
        z = c
        divtime = np.zeros(z.shape, dtype=int)
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == 0)
            divtime[div_now] = i
            z[diverge] = 2
        return torch.from_numpy(divtime).float() / max_iter

    @staticmethod
    def julia(h, w, max_iter=20, c_val=-0.8 + 0.156j):
        # Genera una matriz basada en el conjunto de Julia
        y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
        z = x + y*1j
        divtime = np.zeros(z.shape, dtype=int)
        for i in range(max_iter):
            z = z**2 + c_val
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == 0)
            divtime[div_now] = i
            z[diverge] = 2
        return torch.from_numpy(divtime).float() / max_iter

class AtomoFractal(nn.Module):
    def __init__(self, dim, rank, layers, tipo="mandelbrot", max_seq=5):
        super().__init__()
        gen = GeneradorFractal()
        
        # Pos Emb Fractal
        if tipo == "mandelbrot":
            f_mat = gen.mandelbrot(max_seq, dim)
        else:
            f_mat = gen.julia(max_seq, dim)
        self.pos_emb = nn.Parameter(f_mat.unsqueeze(0) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "A": nn.Linear(dim, rank, bias=False),
                "B": nn.Linear(rank, dim, bias=False)
            }) for _ in range(layers)
        ])
        
        # Inicializar Capas con Rugosidad Fractal
        for i, layer in enumerate(self.layers):
            # Variamos ligeramente las coordenadas para cada capa para que no sean idénticas
            # Pero de forma determinista
            if tipo == "mandelbrot":
                mat_a = gen.mandelbrot(rank, dim, max_iter=20 + i)
                mat_b = gen.mandelbrot(dim, rank, max_iter=20 + i)
            else:
                mat_a = gen.julia(rank, dim, max_iter=20 + i)
                mat_b = gen.julia(dim, rank, max_iter=20 + i)
                
            layer["A"].weight.data.copy_(mat_a * 0.02)
            layer["B"].weight.data.copy_(mat_b * 0.02)
            
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        for layer in self.layers:
            x = x + layer["B"](layer["A"](x))
        return x

def forjar(tipo_fractal="mandelbrot"):
    log_file = f"{FOLDER}/log_ABC_{tipo_fractal}.txt"
    chip_path = f"{FOLDER}/ATOMO_ABC_{tipo_fractal.upper()}.pt"
    
    if os.path.exists(log_file): os.remove(log_file)
    
    def log_print(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_print(f"🔱 INICIANDO FORJA FRACTAL: {tipo_fractal.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map=DEVICE, local_files_only=True)
    embeddings = base_model.get_input_embeddings()
    
    atomo = AtomoFractal(DIM, RANK, LAYERS, tipo=tipo_fractal).to(DEVICE)
    optimizer = optim.Adam(atomo.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, threshold=0.01, verbose=True)
    criterion = nn.MSELoss()

    training_data = []
    for prompt, resp in PARES:
        in_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        out_ids = tokenizer.encode(resp, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        target_vecs = embeddings(out_ids).detach()
        training_data.append((in_ids, target_vecs, out_ids.size(1)))

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
            log_print(f"Epoch {epoch} | Loss: {current_loss:.12f} | Std: {std_loss.item():.12f}")
            torch.save(atomo.state_dict(), chip_path)

    torch.save(atomo.state_dict(), chip_path)
    log_print(f"✅ Forja {tipo_fractal.upper()} Completada.")

if __name__ == "__main__":
    # Podemos elegir el terreno aquí
    forjar("mandelbrot")
