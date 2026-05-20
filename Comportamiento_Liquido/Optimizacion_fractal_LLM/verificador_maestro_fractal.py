# verificador_maestro_fractal.py
# Autor: Andrés Antonio Santisteban Lino
# Sistema Unificado de Auditoría y Comparativa de Relieves Neuronales

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PROTOCOLO 42 Y CONFIGURACIÓN ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Criterios de Éxito de la Auditoría (ALTA COHERENCIA)
TARGET_LOSS = 0.0000909
TARGET_STD = 0.0000000001
LEARNING_RATE = 1e-4

# Dimensiones de la Tríade ABC 24x24
DIM = 896
RANK = 24
LAYERS = 24

LOG_FILE = "log_verificador_maestro.txt"

def log_maestro(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Limpieza inicial del log
import os
if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

# --- 2. CARGA DE COMPONENTES BASE ---
log_maestro(f"Inicializando componentes base en {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEVICE)
embeddings = base_model.get_input_embeddings()
lm_head = base_model.get_output_embeddings() # Necesario para decodificar vectores a palabras

def validar_soberania(modelo, training_data, tokenizer, lm_head):
    """
    Verifica si el modelo entrenado responde correctamente a los prompts.
    Retorna el porcentaje de precisión en la reconstrucción de tokens.
    """
    modelo.eval()
    total_tokens = 0
    tokens_correctos = 0
    
    with torch.no_grad():
        for in_ids, target_vecs, seq_len, target_ids in training_data:
            # Inferencia a través del Átomo
            input_vec = embeddings(in_ids)[:, :1, :].expand(-1, seq_len, -1)
            output_vectors = modelo(input_vec)
            
            # Proyectar a vocabulario y obtener IDs predichos
            logits = lm_head(output_vectors)
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0)
            
            # Comparar con los IDs reales
            actual_ids = target_ids.squeeze(0)
            total_tokens += actual_ids.size(0)
            tokens_correctos += (pred_ids == actual_ids).sum().item()
            
    modelo.train()
    return (tokens_correctos / total_tokens) * 100 if total_tokens > 0 else 0.0

# --- 3. PREPARACIÓN DE DATOS (PARES A, B, C) ---
PARES = [
    ("Hola", "¿Cómo estás?"),     # A
    ("punto", "Hasta luego"),    # B
    ("saber", "No es creer")     # C
]

training_data = []
for prompt, resp in PARES:
    in_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    out_ids = tokenizer.encode(resp, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    target_vecs = embeddings(out_ids).detach()
    # Guardamos también out_ids para la validación de soberanía
    training_data.append((in_ids, target_vecs, out_ids.size(1), out_ids))


# --- 4. MOTOR DE RELIEVE MAESTRO (Génesis Unificado) ---
class MotorRelieveMaestro:
    @staticmethod
    def ruido_estandar(h, w):
        return torch.randn(h, w)
    
    @staticmethod
    def geometrico(h, w):
        v_h = torch.sin(torch.arange(h).float() * 0.5).unsqueeze(1)
        v_w = torch.cos(torch.arange(w).float() * 0.5).unsqueeze(0)
        return torch.mm(v_h, v_w)
    
    @staticmethod
    def mandelbrot(h, w, max_iter=25):
        y, x = np.ogrid[-1.2:1.2:h*1j, -2:1:w*1j]
        c = x + y*1j; z = c
        it = np.zeros(z.shape)
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 4
            it[diverge & (it==0)] = i; z[diverge] = 2
        return torch.from_numpy(it).float() / max_iter


    @staticmethod
    def julia_basilica(h, w, max_iter=25):
        y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
        z = x + y*1j; c = complex(-1.0, 0.0); it = np.zeros(z.shape)
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 4
            it[diverge & (it==0)] = i
        return torch.from_numpy(it).float() / max_iter

    @staticmethod
    def espiral_fractal(h, w, max_iter=25):
        y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
        # c que genera espirales (basado en vórtices continuos y proporción áurea)
        z = x + y*1j; c = complex(-0.8, 0.156); it = np.zeros(z.shape)
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 4
            it[diverge & (it==0)] = i
            z[diverge] = 2 # Prevención de desbordamiento en GPU
        return torch.from_numpy(it).float() / max_iter

    @staticmethod
    def dendrita_fractal(h, w, max_iter=25):
        y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
        # c al borde del caos para fracturar el vacío en canales alienados
        z = x + y*1j; c = complex(-0.7269, 0.1889); it = np.zeros(z.shape)
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 4
            it[diverge & (it==0)] = i
            z[diverge] = 2
        return torch.from_numpy(it).float() / max_iter

    @staticmethod
    def conejo_douady(h, w, max_iter=25):
        y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
        # c del Conejo de Douady: topología orgánica interconectada tipo dendrita pero con lóbulos, lo que reduce el vacío sin caer en caos.
        z = x + y*1j; c = complex(-0.123, 0.745); it = np.zeros(z.shape)
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 4
            it[diverge & (it==0)] = i
            z[diverge] = 2
        return torch.from_numpy(it).float() / max_iter

    @staticmethod
    def espiral_interferencia(h, w, max_iter=25):
        # Combina la espiral áurea con un filtro geométrico para estructurar el vacío
        espiral = MotorRelieveMaestro.espiral_fractal(h, w, max_iter)
        v_h = torch.sin(torch.arange(h).float() * 0.8).unsqueeze(1)
        v_w = torch.cos(torch.arange(w).float() * 0.8).unsqueeze(0)
        grid = torch.abs(torch.mm(v_h, v_w))
        return espiral * grid

# --- 5. ARQUITECTURA DEL ÁTOMO ---

class AtomoEntrenamiento(nn.Module):
    def __init__(self, generador_func):
        super().__init__()
        # Pos Emb inicializado con el relieve seleccionado
        self.pos_emb = nn.Parameter(generador_func(5, DIM).unsqueeze(0) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "A": nn.Linear(DIM, RANK, bias=False),
                "B": nn.Linear(RANK, DIM, bias=False)
            }) for _ in range(LAYERS)
        ])
        
        for layer in self.layers:
            layer["A"].weight.data.copy_(generador_func(RANK, DIM) * 0.02)
            layer["B"].weight.data.copy_(generador_func(DIM, RANK) * 0.02)
            
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        for layer in self.layers:
            x = x + layer["B"](layer["A"](x))
        return x

def guardar_grafico_fractal(modelo, nombre):
    directorio = r"c:\Users\andre\Desktop\Neural_Identity_Forge\EN_DESARROLLO\FISICA_LIQUIDOS\ESTUDIO_FRACTAL\Graficos"
    os.makedirs(directorio, exist_ok=True)
    pesos = modelo.layers[0]["A"].weight.detach().cpu().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(pesos, cmap='viridis', cbar=True)
    plt.title(f"Mapa de Calor de Pesos (Capa 0)\nEstado: {nombre}")
    plt.xlabel("Dimensión de Entrada (896)")
    plt.ylabel("Neuronas Bottleneck (24)")
    plt.subplot(1, 2, 2)
    plt.hist(pesos.flatten(), bins=100, color='darkslateblue', alpha=0.7)
    plt.title("Distribución de Valores\n(Firma de Invariancia de Escala)")
    plt.xlabel("Valor del Peso")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, f"{nombre}.png"), dpi=150)
    plt.close()

# --- 6. SISTEMA DE AUDITORÍA (LEY DE COHERENCIA) ---
def calcular_vacio(modelo):
    total_params = 0
    zeros = 0
    with torch.no_grad():
        for param in modelo.parameters():
            zeros += torch.sum(param == 0).item()
            total_params += param.numel()
    return (zeros / total_params) * 100.0

def realizar_auditoria(modelo, nombre, training_data, embeddings):
    log_maestro(f"\nIniciando entrenamiento del modelo: {nombre}")
    optimizer = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500)
    criterion = nn.MSELoss()
    
    step = 0
    start_time = time.time()
    
    while step < 15000:
        step += 1
        optimizer.zero_grad()
        indiv_losses = []
        
        for in_ids, target_vecs, seq_len, target_ids in training_data:
            input_vec = embeddings(in_ids)[:, :1, :].expand(-1, seq_len, -1)
            output_vectors = modelo(input_vec)
            loss = criterion(output_vectors, target_vecs)
            indiv_losses.append(loss)
        
        # LEY DE COHERENCIA DE ANDRÉS (A, B, C)
        losses_tensor = torch.stack(indiv_losses)
        mean_loss = losses_tensor.mean()
        std_loss = losses_tensor.std() if len(indiv_losses) > 1 else torch.tensor(0.0).to(DEVICE)
        
        total_loss_val = mean_loss + std_loss
        total_loss_val.backward()
        optimizer.step()
        
        current_loss = mean_loss.item()
        current_std = std_loss.item()
        scheduler.step(current_loss)
        
        if current_loss <= TARGET_LOSS and current_std <= TARGET_STD:
            elapsed = time.time() - start_time
            log_maestro(f"Meta alcanzada para {nombre}")
            log_maestro(f"Paso: {step} | Tiempo: {elapsed:.2f}s | Loss: {current_loss:.10f} | Std: {current_std:.12f}")
            return {"pasos": step, "tiempo": elapsed, "loss": current_loss, "std": current_std}
        
        if step == 1 or step % 100 == 0:
            log_maestro(f"[{nombre}] Paso {step} | Loss: {current_loss:.8f} | Std: {current_std:.10f}")

    log_maestro(f"Aviso: {nombre} excedio el limite de entrenamiento.")
    return None

# --- 7. EJECUCIÓN MAESTRA ---
def ejecutar_comparativa_maestra():
    motor = MotorRelieveMaestro()
    contendientes = [
        ("RedEstandar24x24", motor.ruido_estandar),
        ("red_julia", motor.julia_basilica),
        ("RedFractal24x24", motor.mandelbrot),
        ("red_geometrica", motor.geometrico),
        ("RedEspiralFractal", motor.espiral_fractal),
        ("RedDendrita", motor.dendrita_fractal),
        ("RedConejoDouady", motor.conejo_douady),
        ("RedEspiralGrid", motor.espiral_interferencia)
    ]

    resultados_finales = {}
    
    for nombre, gen_func in contendientes:
        # Instanciación limpia para cada carrera
        modelo = AtomoEntrenamiento(gen_func).to(DEVICE)
        
        # Generar y guardar gráfico físico de la inicialización
        guardar_grafico_fractal(modelo, nombre)
        
        # Medir KPI de Vacío (Neuronas muertas en inicialización)
        pct_vacio = calcular_vacio(modelo)
        
        res = realizar_auditoria(modelo, nombre, training_data, embeddings)
        if res:
            res['vacio'] = pct_vacio
            resultados_finales[nombre] = res
        
        # Limpieza de memoria GPU
        del modelo
        torch.cuda.empty_cache()

    # REPORTE FINAL TÉCNICO
    log_maestro("\n" + "="*105)
    log_maestro("REPORTE TÉCNICO FINAL: CARRERA DE SEDIMENTACIÓN FRACTAL (MAESTRO)")
    log_maestro("="*105)
    log_maestro(f"{'Modelo':<22} | {'Pasos':<10} | {'Tiempo':<10} | {'% Vacío':<10} | {'Loss Final':<15} | {'Std Final':<15}")
    log_maestro("-" * 105)
    for nombre, data in resultados_finales.items():
        log_maestro(f"{nombre:<22} | {data['pasos']:<10} | {data['tiempo']:<10.2f}s | {data['vacio']:<8.2f}% | {data['loss']:<15.10f} | {data['std']:<15.12f}")
    log_maestro("="*105)

if __name__ == "__main__":
    ejecutar_comparativa_maestra()
