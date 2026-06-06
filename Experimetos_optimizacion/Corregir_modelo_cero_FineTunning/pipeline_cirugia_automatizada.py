import os
import time
import torch
import subprocess
import json
import random
import numpy as np
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# ==============================================================================
# PIPELINE DE CIRUGIA NEURAL AUTOMATIZADA
# ------------------------------------------------------------------------------
# AUTOR: ANDRES ANTONIO SANTISTEBAN LINO
# UBICACION: EN_DESARROLLO\A_OPTIMIZADOR_RUTAS\Finetunnig_especifico
#
# OBJETIVO:
#   Este script unifica en un solo flujo secuencial dos operaciones que antes
#   se ejecutaban por separado:
#     1. MAPEO DE RUTAS SIR (antes en Busqueda_de_Rutas.py):
#        Calcula el acueducto diferencial de activaciones entre la inferencia
#        basal (modelo crudo) y la inferencia experta (modelo con system message
#        de verdad factica). Esto identifica las neuronas responsables de
#        transportar la informacion factual correcta.
#     2. CIRUGIA QUIRURGICA DE PESOS (antes en cirujano_pesos.py):
#        Congela todo el modelo excepto las neuronas del acueducto SIR,
#        entrena con anclaje L2 para limitar la perturbacion y detiene el
#        entrenamiento al primer paso en que el modelo responde correctamente.
#
# VENTAJA:
#   Al integrarlo en un solo script, el modelo se carga UNA SOLA VEZ en VRAM
#   y el acueducto SIR se calcula en memoria sin escribir archivos intermedios
#   de 40+ MB. Esto reduce el tiempo total y simplifica la operacion.
#
# SALIDA:
#   Todos los archivos se guardan en una subcarpeta con timestamp:
#     YYYYMMDD_HHMM_cirugia/
#       ├── YYYYMMDD_HHMM_acueducto_SIR.json
#       ├── YYYYMMDD_HHMM_log_cirugia.txt
#       ├── YYYYMMDD_HHMM_modelo_cirugia.pt
#       ├── YYYYMMDD_HHMM_evaluacion_pre_cirugia.json
#       ├── YYYYMMDD_HHMM_evaluacion_post_cirugia.json
#       └── YYYYMMDD_HHMM_comparativa_cirugia.json
# ==============================================================================

# --- ENRUTAMIENTO DINAMICO ---
# Resolvemos las rutas relativas a partir de la ubicacion de este script,
# subiendo 3 niveles hasta la raiz del proyecto (Neural_Identity_Forge).
# Esto permite que el script funcione independientemente de donde se ejecute.
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, "..", "..", ".."))
PROTOCOLO_MAESTRO = os.path.join(BASE_DIR, "ADN_RAW", "protocolo_maestro_laboratorio.json")

# Seleccion automatica del dispositivo de computo.
# Se usa CUDA si hay GPU disponible, de lo contrario CPU (mucho mas lento).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HIPERPARAMETROS DE CIRUGIA ---
# Estos valores fueron calibrados experimentalmente en las versiones v1-v5
# del cirujano de pesos para lograr convergencia sin olvido catastrofico.
EPOCHS = 100                  # Limite maximo de iteraciones (red de seguridad)
LEARNING_RATE = 1e-5          # Tasa de aprendizaje suave: al concentrar la fuerza
                              # en pocas neuronas, un LR alto causa colapso total.
                              # 1e-5 permite movimientos infinitesimales.
PASOS_VISTA_PREVIA = 10       # Cada cuantos epochs generar una vista previa con
                              # una pregunta de control aleatoria para monitorizar
                              # el olvido catastrofico en tiempo real.
LOSS_DIVERGENCIA = 15.0       # Si la loss supera este umbral, se aborta el
                              # entrenamiento para proteger los pesos del modelo.
ALPHA_ANCLAJE = 1.0           # Peso de la regularizacion L2 de anclaje. Un valor
                              # alto (1.0) actua como un muelle elastico que impide
                              # que los pesos se alejen demasiado de su estado virgen.
                              # Matematicamente: L_total = L_ce + alpha * ||W - W0||^2
TOP_N_NEURONAS = 10           # Numero de neuronas maestras por capa a descongelar.
                              # Limitar a 10 reduce el subespacio de optimizacion
                              # a <1% de los parametros totales, evitando el colapso
                              # que ocurria con la mascara completa (v3).
MAX_NEW_TOKENS_EVAL = 64      # Tokens maximos que el modelo puede generar en
                              # cada evaluacion. 64 es suficiente para respuestas
                              # cortas y factuales sin consumir VRAM excesiva.

# --- CASO DE CIRUGIA ---
# Define el concepto erroneo a corregir, la pregunta basal que lo activa,
# la respuesta correcta que se inyectara, y el system message experto que
# se usara para calcular el acueducto SIR diferencial.
CASO_CIRUGIA = {
    "concepto": "ANATOMÍA_CABALLO",
    "pregunta": "¿Cuántas patas tiene un caballo?",
    "respuesta_target": "Un caballo tiene 4 patas.",
    "verdad_sistema": "Eres un experto en zoología equina y la respuesta correcta es que un caballo tiene 4 patas."
}

# --- BANCO DE 30 PREGUNTAS DE CONTROL ---
# Estas 30 preguntas sirven como "red de seguridad" para detectar olvido
# catastrofico. Se ejecutan ANTES y DESPUES de la cirugia para comparar
# si la correccion del caballo ha degradado otros conocimientos del modelo.
# Se dividen en 4 bloques:
#   A) Casos de falla originales (7): preguntas que el modelo base falla
#   B) Animales y naturaleza (8): conocimiento general cercano al concepto
#   C) Numeros y colores (8): conocimiento factual basico
#   D) Cultura basica (7): geografia, idiomas, cuerpo humano
BANCO_CONTROL = [
    # Bloque A: Casos de Falla Originales (7)
    {"id": 1,  "concepto": "ANATOMÍA_GATO",        "pregunta": "¿Cuántas patas tiene un gato?"},
    {"id": 2,  "concepto": "ANATOMÍA_CABALLO",      "pregunta": "¿Cuántas patas tiene un caballo?"},
    {"id": 3,  "concepto": "LOGICA_CHALECO",        "pregunta": "¿De qué color son las mangas del chaleco rojo de Sebastián?"},
    {"id": 4,  "concepto": "FISICA_PESO",           "pregunta": "¿Qué pesa más, un kilo de plumas o dos kilos de hierro?"},
    {"id": 5,  "concepto": "NEGOCIOS_PUNTO_EQUILIBRIO", "pregunta": "¿Cómo se calcula el punto de equilibrio en unidades para un negocio?"},
    {"id": 6,  "concepto": "NEGOCIOS_ROI",          "pregunta": "¿Cuál es la fórmula para obtener el ROI en marketing?"},
    {"id": 7,  "concepto": "ASTRONOMÍA_TIERRA",     "pregunta": "Verdad o mentira: La Tierra tarda 24 horas en dar una vuelta completa alrededor del Sol."},
    # Bloque B: Animales y Naturaleza (8)
    {"id": 8,  "concepto": "ANIMAL_PERRO",          "pregunta": "¿Cuántas patas tiene un perro?"},
    {"id": 9,  "concepto": "ANIMAL_GALLINA",        "pregunta": "¿Cuántas patas tiene una gallina?"},
    {"id": 10, "concepto": "ANIMAL_ARANA",          "pregunta": "¿Cuántas patas tiene una araña?"},
    {"id": 11, "concepto": "ANIMAL_PEZ",            "pregunta": "¿Los peces viven en el agua o en la tierra?"},
    {"id": 12, "concepto": "ANIMAL_PAJARO",         "pregunta": "¿Los pájaros pueden volar?"},
    {"id": 13, "concepto": "NATURALEZA_SOL",        "pregunta": "¿El Sol es una estrella o un planeta?"},
    {"id": 14, "concepto": "NATURALEZA_AGUA",       "pregunta": "¿El agua es líquida, sólida o gaseosa a temperatura ambiente?"},
    {"id": 15, "concepto": "NATURALEZA_LUNA",       "pregunta": "¿La Luna brilla con luz propia?"},
    # Bloque C: Numeros y Colores (8)
    {"id": 16, "concepto": "NUMERO_SUMA",           "pregunta": "¿Cuánto es 2 + 2?"},
    {"id": 17, "concepto": "NUMERO_MULTI",          "pregunta": "¿Cuánto es 3 por 3?"},
    {"id": 18, "concepto": "NUMERO_DIAS",           "pregunta": "¿Cuántos días tiene una semana?"},
    {"id": 19, "concepto": "NUMERO_MESES",          "pregunta": "¿Cuántos meses tiene un año?"},
    {"id": 20, "concepto": "NUMERO_HORAS",          "pregunta": "¿Cuántas horas tiene un día?"},
    {"id": 21, "concepto": "COLOR_CIELO",           "pregunta": "¿De qué color es el cielo en un día despejado?"},
    {"id": 22, "concepto": "COLOR_SANGRE",          "pregunta": "¿De qué color es la sangre?"},
    {"id": 23, "concepto": "COLOR_NIEVE",           "pregunta": "¿De qué color es la nieve?"},
    # Bloque D: Cultura Basica (7)
    {"id": 24, "concepto": "GEO_OCEANO",            "pregunta": "¿Cuál es el océano más grande del mundo?"},
    {"id": 25, "concepto": "GEO_CONTINENTE",        "pregunta": "¿En qué continente está España?"},
    {"id": 26, "concepto": "IDIOMA_BRASIL",         "pregunta": "¿Qué idioma se habla en Brasil?"},
    {"id": 27, "concepto": "DEPORTE_FUTBOL",        "pregunta": "¿Cuántos jugadores tiene un equipo de fútbol en el campo?"},
    {"id": 28, "concepto": "CUERPO_OJOS",           "pregunta": "¿Cuántos ojos tiene una persona?"},
    {"id": 29, "concepto": "CUERPO_DEDOS",          "pregunta": "¿Cuántos dedos tiene una mano?"},
    {"id": 30, "concepto": "FRUTA_PLATANO",         "pregunta": "¿De qué color es un plátano maduro?"},
]


# =============================================================================
# SISTEMA DE LOGS EN VIVO
# -----------------------------------------------------------------------------
# Todas las operaciones del pipeline se registran simultaneamente en:
#   1. La consola (stdout) para monitoreo en tiempo real
#   2. Un archivo de texto dentro de la subcarpeta de salida
# Esto permite auditar el experimento completo despues de su ejecucion.
# =============================================================================
ARCHIVO_LOG = None

def log_info(mensaje, imprimir_consola=True):
    """Registra un mensaje con timestamp en consola y archivo de log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mensaje_formateado = f"[{timestamp}] {mensaje}"
    
    if imprimir_consola:
        print(mensaje_formateado)
        
    if ARCHIVO_LOG:
        with open(ARCHIVO_LOG, "a", encoding="utf-8") as f:
            f.write(mensaje_formateado + "\n")


# =============================================================================
# ESCUDO TERMICO DE LA GPU
# -----------------------------------------------------------------------------
# Proteccion contra sobrecalentamiento durante operaciones prolongadas.
# Si la GPU supera max_temp, el script se pausa automaticamente hasta que
# la temperatura descienda a temp_segura. Esto previene throttling termico
# y posibles danos al hardware durante el entrenamiento en bucle.
# =============================================================================
def termostato_gpu(max_temp=85, temp_segura=60, pausa_segundos=10):
    """Monitorea la temperatura de la GPU y pausa si supera el umbral."""
    try:
        resultado = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], 
            stdout=subprocess.PIPE, text=True
        )
        temp_actual = int(resultado.stdout.strip().split('\n')[0])
        
        if temp_actual >= max_temp:
            log_info(f"[ESCUDO TERMICO] GPU a {temp_actual} C. Esperando enfriamiento a {temp_segura} C...")
            while temp_actual > temp_segura:
                time.sleep(pausa_segundos)
                resultado = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], 
                    stdout=subprocess.PIPE, text=True
                )
                temp_actual = int(resultado.stdout.strip().split('\n')[0])
            log_info("[ESCUDO TERMICO] GPU segura. Reanudando proceso.")
    except Exception:
        # Si nvidia-smi no esta disponible (ej. sin GPU), ignoramos silenciosamente
        pass 


# =============================================================================
# FASE 1: MAPEO DINAMICO DE RUTAS SIR
# -----------------------------------------------------------------------------
# SIR = System-Induced Routing (Enrutamiento Inducido por Sistema)
#
# PRINCIPIO TEORICO:
# Cuando inyectamos la verdad factica en el system message, el modelo activa
# un canal interno diferente al que usa cuando responde sin guia. La diferencia
# entre ambas activaciones revela las neuronas MLP responsables de "transportar"
# la informacion factual correcta. Este diferencial se denomina "acueducto SIR".
#
# PROCESO:
# 1. Se instalan hooks de PyTorch (register_forward_hook) en la salida de
#    cada bloque MLP del modelo. Estos hooks capturan la activacion media
#    (promediada sobre la dimension de secuencia) sin interferir con el
#    computo normal del modelo.
# 2. Se ejecuta un forward pass con el prompt BASAL (sin system message)
#    y se registra h_basal[l] para cada capa l.
# 3. Se ejecuta un forward pass con el prompt EXPERTO (con system message
#    de verdad factica) y se registra h_experto[l].
# 4. Se calcula el vector diferencial: v[l] = h_experto[l] - h_basal[l]
# 5. Se seleccionan las top_n neuronas con mayor |v[l]| por capa.
#    Estas son las "neuronas maestras" del acueducto SIR.
# 6. Se desinstalan los hooks (ya no se necesitan para el entrenamiento).
# =============================================================================
def mapear_acueducto_sir(model, tokenizer, caso, top_n=TOP_N_NEURONAS):
    """
    Calcula el acueducto SIR en memoria sin escribir archivos intermedios.
    
    Argumentos:
        model: modelo cargado en VRAM
        tokenizer: tokenizer del modelo
        caso: dict con pregunta, verdad_sistema, etc.
        top_n: cuantas neuronas maestras seleccionar por capa
    
    Retorna:
        acueducto_data: lista con informacion del diferencial por capa (para JSON)
        mascaras_por_capa: dict {capa_idx: [indices_neuronas]} (para la cirugia)
        resp_basal: respuesta del modelo sin system message
        resp_experto: respuesta del modelo con system message de verdad factica
    """
    log_info("\n[FASE 1] Iniciando mapeo dinamico de rutas SIR...")
    model.eval()
    
    # Diccionario temporal para almacenar las activaciones capturadas por los hooks.
    # Se limpia antes de cada forward pass para evitar contaminacion cruzada.
    activaciones = {}
    
    def crear_hook(layer_idx, etiqueta):
        """
        Fabrica de hooks de reenvio (forward hooks).
        Cada hook captura la activacion media de la salida del bloque MLP
        y la almacena en el diccionario 'activaciones' bajo la clave
        (layer_idx, etiqueta). Se usa mean(dim=1) para promediar sobre
        la dimension de secuencia, obteniendo un vector de hidden_size
        que representa la "presion" media de cada neurona en esa capa.
        """
        def hook(module, input, output):
            # El output de un MLP puede ser un tuple; tomamos el tensor principal
            tensor = output[0] if isinstance(output, tuple) else output
            # Promediar sobre la dimension de secuencia (dim=1) para obtener
            # un unico vector de activacion representativo de toda la secuencia
            activacion_media = tensor.detach().cpu().mean(dim=1).squeeze()
            if layer_idx not in activaciones:
                activaciones[layer_idx] = {}
            activaciones[layer_idx][etiqueta] = activacion_media
        return hook
    
    # Instalar hooks temporales en la salida de cada bloque MLP.
    # Se registran en TODAS las capas del transformer para obtener
    # un mapa completo del flujo de informacion factual.
    hooks = []
    num_capas = len(model.model.layers)
    for i in range(num_capas):
        h = model.model.layers[i].mlp.register_forward_hook(crear_hook(i, "actual"))
        hooks.append(h)
    
    log_info(f"   Hooks de activacion instalados en {num_capas} capas MLP.")
    
    # --- FORWARD PASS BASAL ---
    # Ejecutamos la pregunta SIN system message. Esto representa el comportamiento
    # "natural" del modelo, que en este caso responde incorrectamente ("2 patas").
    # Las activaciones capturadas reflejan el patron de procesamiento erroneo.
    termostato_gpu()
    conversacion_basal = [{"role": "user", "content": caso["pregunta"]}]
    ids_basal = tokenizer.apply_chat_template(
        conversacion_basal, tokenize=True, 
        add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)
    
    activaciones.clear()
    with torch.no_grad():
        model(ids_basal)
    
    # Copiar las activaciones basales antes de limpiar el diccionario
    act_basal = {}
    for layer_idx in range(num_capas):
        if layer_idx in activaciones and "actual" in activaciones[layer_idx]:
            act_basal[layer_idx] = activaciones[layer_idx]["actual"].clone()
        else:
            # Si por alguna razon una capa no registro activacion, usar ceros
            act_basal[layer_idx] = torch.zeros(model.config.hidden_size)
    
    # Generar la respuesta basal completa para registrarla en el log.
    # Usamos do_sample=False (Greedy Search) para garantizar determinismo.
    activaciones.clear()
    with torch.no_grad():
        out_basal = model.generate(ids_basal, max_new_tokens=64, do_sample=False)
    resp_basal = tokenizer.decode(out_basal[0][ids_basal.shape[1]:], skip_special_tokens=True).strip()
    log_info(f"   Respuesta BASAL: '{resp_basal}'")
    
    # --- FORWARD PASS EXPERTO ---
    # Ejecutamos la misma pregunta PERO con un system message que contiene
    # la verdad factica. Esto fuerza al modelo a activar un canal interno
    # diferente que transporta la respuesta correcta ("4 patas").
    # La diferencia entre ambos patrones de activacion revela el acueducto.
    termostato_gpu()
    conversacion_sir = [
        {"role": "system", "content": caso["verdad_sistema"]},
        {"role": "user", "content": caso["pregunta"]}
    ]
    ids_sir = tokenizer.apply_chat_template(
        conversacion_sir, tokenize=True, 
        add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)
    
    activaciones.clear()
    with torch.no_grad():
        model(ids_sir)
    
    # Copiar las activaciones expertas
    act_experto = {}
    for layer_idx in range(num_capas):
        if layer_idx in activaciones and "actual" in activaciones[layer_idx]:
            act_experto[layer_idx] = activaciones[layer_idx]["actual"].clone()
        else:
            act_experto[layer_idx] = torch.zeros(model.config.hidden_size)
    
    # Generar la respuesta experta para el log
    activaciones.clear()
    with torch.no_grad():
        out_sir = model.generate(ids_sir, max_new_tokens=64, do_sample=False)
    resp_experto = tokenizer.decode(out_sir[0][ids_sir.shape[1]:], skip_special_tokens=True).strip()
    log_info(f"   Respuesta EXPERTO: '{resp_experto}'")
    
    # Desinstalar hooks de activacion. Ya hemos capturado todo lo necesario
    # y mantener los hooks activos durante el entrenamiento consumiria memoria
    # innecesariamente y podria interferir con el computo de gradientes.
    for h in hooks:
        h.remove()
    log_info(f"   Hooks de activacion desinstalados.")
    
    # --- CALCULO DEL DIFERENCIAL SIR Y SELECCION DE NEURONAS MAESTRAS ---
    # Para cada capa, calculamos v[l] = h_experto[l] - h_basal[l].
    # Este vector indica CUANTO cambia cada neurona cuando el modelo pasa
    # de "no saber" (basal) a "saber" (experto). Las neuronas con mayor
    # magnitud absoluta |v[l,i]| son las que mas contribuyen al transporte
    # de la informacion factual correcta y se seleccionan como "maestras".
    hidden_size = model.config.hidden_size
    acueducto_data = []
    mascaras_por_capa = {}
    
    for layer_idx in range(num_capas):
        basal = act_basal[layer_idx]
        experto = act_experto[layer_idx]
        
        # Diferencial SIR: v_l = h_experto - h_basal
        # Un valor positivo alto indica que esa neurona se activa MAS
        # cuando el modelo "sabe" la respuesta correcta.
        delta_sir = experto - basal
        abs_delta = torch.abs(delta_sir)
        
        # Seleccionar las top_n neuronas con mayor activacion diferencial.
        # Usamos torch.topk para eficiencia en lugar de ordenar todo el vector.
        if abs_delta.numel() >= top_n:
            _, indices_top = torch.topk(abs_delta, top_n)
        else:
            indices_top = torch.arange(abs_delta.numel())
        
        # Guardar informacion de las neuronas maestras para el JSON
        neuronas_maestras = [
            {"idx": idx.item(), "val": delta_sir[idx].item()}
            for idx in indices_top
        ]
        
        # Filtrar indices que caigan dentro del espacio hidden valido.
        # Esto es una precaucion por si el modelo tiene dimensiones atipicas.
        indices_validos = [idx.item() for idx in indices_top if idx.item() < hidden_size]
        mascaras_por_capa[layer_idx] = sorted(indices_validos)
        
        acueducto_data.append({
            "capa": layer_idx,
            "delta_sir_norm": torch.norm(delta_sir).item(),
            "neuronas_maestras": neuronas_maestras
        })
    
    log_info(f"   Acueducto SIR calculado: {num_capas} capas, top {top_n} neuronas por capa.")
    log_info(f"[FASE 1] Mapeo SIR completado.\n")
    
    return acueducto_data, mascaras_por_capa, resp_basal, resp_experto


# =============================================================================
# CONGELAMIENTO SELECTIVO DEL MODELO
# -----------------------------------------------------------------------------
# PRINCIPIO TEORICO:
# PyTorch no permite congelar filas individuales de un tensor de pesos.
# Para restringir el gradiente a las neuronas del acueducto, usamos una
# estrategia de dos pasos:
#   1. Descongelamos las matrices MLP completas de las capas afectadas
#      (gate_proj, up_proj, down_proj).
#   2. Instalamos "gradient hooks" (register_hook) que multiplican el
#      gradiente por una mascara binaria ANTES de que el optimizador
#      lo aplique. Las neuronas fuera del acueducto reciben gradiente = 0,
#      por lo que sus pesos no cambian aunque esten tecnicamente "descongelados".
#
# ANATOMIA DE UN BLOQUE MLP EN QWEN:
#   gate_proj: [intermediate_size x hidden_size] -> Compuerta de activacion
#   up_proj:   [intermediate_size x hidden_size] -> Proyeccion de expansion
#   down_proj: [hidden_size x intermediate_size] -> Proyeccion de compresion
#
# Las neuronas del acueducto estan indexadas en el espacio hidden_size,
# por lo que la mascara se aplica sobre COLUMNAS de gate/up y FILAS de down.
# =============================================================================
def aplicar_congelamiento_quirurgico(model, mascaras_por_capa):
    """
    Congela todo el modelo y descongela selectivamente solo las neuronas
    del acueducto SIR, instalando hooks de gradiente para enmascarar
    las actualizaciones en las neuronas no seleccionadas.
    
    Retorna la lista de hooks para poder limpiarlos al finalizar.
    """
    # Paso 1: Congelar ABSOLUTAMENTE todo el modelo.
    # Ningun parametro recibira gradiente por defecto.
    model.requires_grad_(False)
    
    parametros_descongelados = 0
    parametros_totales = sum(p.numel() for p in model.parameters())
    hooks_gradiente = []
    
    for capa_idx, indices_neuronas in mascaras_por_capa.items():
        if not indices_neuronas:
            continue
            
        mlp = model.model.layers[capa_idx].mlp
        hidden_size = mlp.down_proj.weight.shape[0]
        
        # Paso 2: Descongelar las 3 matrices MLP de esta capa.
        # Aunque las descongelamos completas, los hooks de gradiente
        # se encargaran de anular el gradiente en las filas/columnas
        # que NO pertenecen al acueducto.
        mlp.gate_proj.weight.requires_grad_(True)
        mlp.up_proj.weight.requires_grad_(True)
        mlp.down_proj.weight.requires_grad_(True)
        
        parametros_descongelados += mlp.gate_proj.weight.numel()
        parametros_descongelados += mlp.up_proj.weight.numel()
        parametros_descongelados += mlp.down_proj.weight.numel()
        
        # Paso 3: Construir mascara binaria.
        # Un vector de ceros con 1.0 solo en las posiciones de las neuronas
        # seleccionadas por el acueducto SIR.
        mascara_hidden = torch.zeros(hidden_size, device=DEVICE, dtype=torch.bfloat16)
        for idx in indices_neuronas:
            if idx < hidden_size:
                mascara_hidden[idx] = 1.0
        
        # Paso 4: Crear hooks de gradiente.
        # Estos hooks interceptan el gradiente ANTES de que el optimizador
        # lo use, multiplicandolo por la mascara para anular las columnas/filas
        # no pertenecientes al acueducto.
        
        # Para gate_proj y up_proj (shape: [intermediate_size, hidden_size]):
        # La mascara se aplica sobre la dimension de COLUMNAS (hidden_size).
        # unsqueeze(0) convierte el vector [hidden_size] en [1, hidden_size]
        # para que el broadcasting funcione correctamente.
        def crear_hook_gate_up(mascara_h):
            def hook(grad):
                return grad * mascara_h.unsqueeze(0)
            return hook
        
        # Para down_proj (shape: [hidden_size, intermediate_size]):
        # La mascara se aplica sobre la dimension de FILAS (hidden_size).
        # unsqueeze(1) convierte el vector [hidden_size] en [hidden_size, 1].
        def crear_hook_down(mascara_h):
            def hook(grad):
                return grad * mascara_h.unsqueeze(1)
            return hook
        
        h1 = mlp.gate_proj.weight.register_hook(crear_hook_gate_up(mascara_hidden))
        h2 = mlp.up_proj.weight.register_hook(crear_hook_gate_up(mascara_hidden))
        h3 = mlp.down_proj.weight.register_hook(crear_hook_down(mascara_hidden))
        hooks_gradiente.extend([h1, h2, h3])
    
    ratio = (parametros_descongelados / parametros_totales) * 100
    log_info(f"[CIRUGIA] Parametros totales: {parametros_totales:,}")
    log_info(f"[CIRUGIA] Parametros descongelados (brutos): {parametros_descongelados:,} ({ratio:.2f}%)")
    log_info(f"[CIRUGIA] Capas intervenidas: {len(mascaras_por_capa)}")
    log_info(f"[CIRUGIA] Hooks de mascara de gradiente instalados: {len(hooks_gradiente)}")
    
    return hooks_gradiente


# =============================================================================
# GENERACION DE RESPUESTA (INFERENCIA GREEDY DETERMINISTA)
# -----------------------------------------------------------------------------
# Genera una respuesta del modelo usando decodificacion Greedy (do_sample=False).
# Esto equivale matematicamente a temperatura T=0:
#   y_t = argmax_w P(w | y_<t, x)
# En cada paso de decodificacion se selecciona el token de maxima probabilidad,
# eliminando toda estocasticidad. Esto garantiza que la misma entrada produzca
# siempre la misma salida, lo cual es critico para:
#   1. Evaluar de forma reproducible la convergencia durante el entrenamiento
#   2. Comparar respuestas PRE vs POST de forma justa
# =============================================================================
def generar_respuesta(model, tokenizer, pregunta, max_tokens=MAX_NEW_TOKENS_EVAL):
    """Genera una respuesta determinista Greedy para una pregunta dada."""
    model.eval()
    conversacion = [{"role": "user", "content": pregunta}]
    ids = tokenizer.apply_chat_template(
        conversacion, tokenize=True, 
        add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
    
    # Decodificar solo los tokens generados (excluyendo el prompt de entrada)
    respuesta = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    model.train()
    return respuesta


# =============================================================================
# EVALUACION COMPLETA DE 30 PREGUNTAS DE CONTROL
# -----------------------------------------------------------------------------
# Ejecuta todas las preguntas del banco de control y registra las respuestas.
# Se usa tanto ANTES (PRE) como DESPUES (POST) de la cirugia para medir
# el impacto de la modificacion de pesos en el conocimiento general del modelo.
# =============================================================================
def evaluar_banco_completo(model, tokenizer, fase="PRE"):
    """Evalua las 30 preguntas de control y retorna los resultados."""
    log_info(f"\n{'='*60}")
    log_info(f"  EVALUACION {fase}-CIRUGIA: 30 PREGUNTAS DE CONTROL")
    log_info(f"{'='*60}")
    
    resultados = []
    
    for i, caso in enumerate(BANCO_CONTROL, 1):
        # Proteccion termica antes de cada inferencia
        termostato_gpu(max_temp=85, temp_segura=60)
        
        respuesta = generar_respuesta(model, tokenizer, caso["pregunta"])
        
        resultados.append({
            "id": caso["id"],
            "concepto": caso["concepto"],
            "pregunta": caso["pregunta"],
            "respuesta": respuesta
        })
        
        # Mostrar resumen compacto en consola (truncado a 60 caracteres)
        resp_corta = respuesta[:60].replace('\n', ' ')
        log_info(f"  [{i:02d}/30] {caso['concepto']}: '{resp_corta}...'")
    
    log_info(f"{'='*60}\n")
    return resultados


# =============================================================================
# VISTA PREVIA EN VIVO (DURANTE ENTRENAMIENTO)
# -----------------------------------------------------------------------------
# Genera una vista previa rapida durante el entrenamiento que incluye:
#   1. La respuesta actual a la pregunta objetivo (para ver si ya convergio)
#   2. Una pregunta ALEATORIA del banco de control (para detectar olvido
#      catastrofico en tiempo real sin tener que esperar a la evaluacion POST)
# =============================================================================
def generar_vista_previa(model, tokenizer, pregunta_objetivo, banco_control, epoch=0):
    """Genera vista previa con la pregunta objetivo y una de control aleatoria."""
    model.eval()
    
    log_info("\n--- VISTA PREVIA EN VIVO ---")
    
    # 1. Pregunta objetivo de la cirugia
    resp_objetivo = generar_respuesta(model, tokenizer, pregunta_objetivo)
    log_info(f"  [OBJETIVO] '{pregunta_objetivo}'")
    log_info(f"  [RESPUESTA] '{resp_objetivo}'")
    
    # 2. Pregunta aleatoria del banco de control (canario en la mina)
    caso_aleatorio = random.choice(banco_control)
    resp_control = generar_respuesta(model, tokenizer, caso_aleatorio["pregunta"])
    log_info(f"  [CONTROL ALEATORIO] ({caso_aleatorio['concepto']}) '{caso_aleatorio['pregunta']}'")
    log_info(f"  [RESPUESTA] '{resp_control}'")
    
    log_info("--- FIN VISTA PREVIA ---\n")
    
    model.train()
    
    return {
        "epoch": epoch,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "objetivo": {
            "pregunta": pregunta_objetivo,
            "respuesta": resp_objetivo
        },
        "control_aleatorio": {
            "concepto": caso_aleatorio["concepto"],
            "pregunta": caso_aleatorio["pregunta"],
            "respuesta": resp_control
        }
    }


# =============================================================================
# MOTOR PRINCIPAL DEL PIPELINE
# =============================================================================
def main():
    global ARCHIVO_LOG
    
    # Fijar semilla para reproducibilidad total del experimento.
    # Se fija en Python (random), NumPy y PyTorch (CPU y GPU) para
    # garantizar que los mismos inputs produzcan los mismos outputs
    # en cada ejecucion, facilitando la comparacion entre experimentos.
    semilla = 42
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    
    # =========================================================================
    # FASE 0: INICIALIZACION Y ENRUTAMIENTO
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Crear subcarpeta unica para esta ejecucion del pipeline.
    # El nombre incluye fecha y hora para evitar colisiones y permitir
    # multiples ejecuciones sin sobreescribir resultados anteriores.
    carpeta_salida = os.path.join(PIPELINE_DIR, f"{timestamp}_cirugia")
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Definir las rutas de todos los archivos de salida dentro de la subcarpeta
    ARCHIVO_LOG = os.path.join(carpeta_salida, f"{timestamp}_log_cirugia.txt")
    ruta_acueducto = os.path.join(carpeta_salida, f"{timestamp}_acueducto_SIR.json")
    ruta_modelo_pt = os.path.join(carpeta_salida, f"{timestamp}_modelo_cirugia.pt")
    ruta_eval_pre = os.path.join(carpeta_salida, f"{timestamp}_evaluacion_pre_cirugia.json")
    ruta_eval_post = os.path.join(carpeta_salida, f"{timestamp}_evaluacion_post_cirugia.json")
    ruta_comparativa = os.path.join(carpeta_salida, f"{timestamp}_comparativa_cirugia.json")
    
    # Iniciar el archivo de log
    with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
        f.write(f"--- PIPELINE DE CIRUGIA NEURAL AUTOMATIZADA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    
    log_info("=" * 60)
    log_info("  PIPELINE DE CIRUGIA NEURAL AUTOMATIZADA")
    log_info("  Autor: Andres Antonio Santisteban Lino")
    log_info("=" * 60)
    log_info(f"[FASE 0] Subcarpeta de salida: {os.path.basename(carpeta_salida)}/")
    
    # Cargar el protocolo maestro del laboratorio, que contiene la ruta
    # al modelo Qwen local y otros parametros de configuracion.
    if not os.path.exists(PROTOCOLO_MAESTRO):
        log_info(f"ERROR: No se encuentra el protocolo maestro en {PROTOCOLO_MAESTRO}")
        return
    
    with open(PROTOCOLO_MAESTRO, "r", encoding="utf-8") as f:
        maestro = json.load(f)
    
    model_id = maestro["parameters"]["model_id"]
    if not os.path.isabs(model_id):
        model_id = os.path.abspath(os.path.join(BASE_DIR, model_id))
    
    log_info(f"[FASE 0] Modelo: {os.path.basename(model_id)}")
    log_info(f"[FASE 0] Concepto objetivo: {CASO_CIRUGIA['concepto']}")
    log_info(f"[FASE 0] Pregunta: {CASO_CIRUGIA['pregunta']}")
    log_info(f"[FASE 0] Target: {CASO_CIRUGIA['respuesta_target']}")
    
    # Cargar el modelo UNA SOLA VEZ en bfloat16.
    # Usamos bfloat16 (en lugar de float16) porque tiene el mismo rango
    # dinamico que float32 (8 bits de exponente) evitando los desbordamientos
    # de gradiente (NaN) que ocurrieron en la version v2 con float16.
    termostato_gpu()
    log_info("[FASE 0] Cargando reactor neural (bfloat16)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(DEVICE)
    
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    log_info(f"[FASE 0] Hidden size: {hidden_size}, Intermediate size: {intermediate_size}")
    log_info(f"[FASE 0] Inicializacion completada.\n")
    
    tiempo_inicio_total = time.time()
    
    # =========================================================================
    # FASE 1: MAPEO DINAMICO DE RUTAS SIR
    # Se calcula el acueducto diferencial en memoria y se construye la mascara
    # de gradientes que se usara en la cirugia.
    # =========================================================================
    acueducto_data, mascaras_por_capa, resp_basal, resp_experto = mapear_acueducto_sir(
        model, tokenizer, CASO_CIRUGIA, top_n=TOP_N_NEURONAS
    )
    
    # Persistir el acueducto calculado en la subcarpeta para auditoria
    informe_acueducto = {
        "metadatos": {
            "autor": "Andres Antonio Santisteban Lino",
            "timestamp": timestamp,
            "modelo": os.path.basename(model_id),
            "concepto": CASO_CIRUGIA["concepto"],
            "top_n_neuronas": TOP_N_NEURONAS
        },
        "respuesta_basal": resp_basal,
        "respuesta_experto": resp_experto,
        "acueducto": acueducto_data
    }
    with open(ruta_acueducto, "w", encoding="utf-8") as f:
        json.dump(informe_acueducto, f, indent=4, ensure_ascii=False)
    log_info(f"[FASE 1] Acueducto SIR guardado en: {os.path.basename(ruta_acueducto)}")
    
    # =========================================================================
    # FASE 2: EVALUACION PRE-CIRUGIA
    # Registra las respuestas del modelo VIRGEN a las 30 preguntas de control.
    # Esto establece la linea base para la comparacion posterior.
    # =========================================================================
    log_info("\n[FASE 2] Iniciando evaluacion PRE-cirugia...")
    model.eval()
    resultados_pre = evaluar_banco_completo(model, tokenizer, fase="PRE")
    
    informe_pre = {
        "metadatos": {
            "autor": "Andres Antonio Santisteban Lino",
            "timestamp": timestamp,
            "fase": "PRE-CIRUGIA",
            "modelo": os.path.basename(model_id)
        },
        "resultados": resultados_pre
    }
    with open(ruta_eval_pre, "w", encoding="utf-8") as f:
        json.dump(informe_pre, f, indent=4, ensure_ascii=False)
    log_info(f"[FASE 2] Evaluacion PRE guardada en: {os.path.basename(ruta_eval_pre)}")
    
    # =========================================================================
    # FASE 3: CONGELAMIENTO QUIRURGICO
    # Se congela todo el modelo y se descongela selectivamente solo las
    # neuronas del acueducto SIR para la optimizacion restringida.
    # =========================================================================
    log_info("\n[FASE 3] Preparando cirugia quirurgica...")
    hooks_gradiente = aplicar_congelamiento_quirurgico(model, mascaras_por_capa)
    
    # Construir la secuencia de entrenamiento: prompt + respuesta correcta.
    # El modelo aprendera a generar esta respuesta cuando reciba el prompt basal.
    conversacion_target = [
        {"role": "user", "content": CASO_CIRUGIA["pregunta"]},
        {"role": "assistant", "content": CASO_CIRUGIA["respuesta_target"]}
    ]
    
    ids_completos = tokenizer.apply_chat_template(
        conversacion_target, tokenize=True,
        add_generation_prompt=False, return_tensors="pt"
    ).to(DEVICE)
    
    # Tokenizar solo el prompt para calcular donde empieza la respuesta.
    # Los tokens del prompt se enmascaran con -100 en los labels para que
    # NO contribuyan a la cross-entropy loss. Solo los tokens de la respuesta
    # target generan gradiente, obligando al modelo a aprender UNICAMENTE
    # la respuesta correcta sin "desaprender" la estructura del prompt.
    conversacion_solo_prompt = [
        {"role": "user", "content": CASO_CIRUGIA["pregunta"]}
    ]
    ids_prompt = tokenizer.apply_chat_template(
        conversacion_solo_prompt, tokenize=True,
        add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)
    
    longitud_prompt = ids_prompt.shape[1]
    longitud_total = ids_completos.shape[1]
    
    log_info(f"[CIRUGIA] Tokens del prompt: {longitud_prompt}")
    log_info(f"[CIRUGIA] Tokens totales (prompt + target): {longitud_total}")
    log_info(f"[CIRUGIA] Tokens de la respuesta target: {longitud_total - longitud_prompt}")
    
    # Crear labels con -100 en la region del prompt (ignorados por cross-entropy)
    labels = ids_completos.clone()
    labels[0, :longitud_prompt] = -100
    
    # =========================================================================
    # FASE 4: DESCENSO DE GRADIENTE RESTRINGIDO
    # Bucle de entrenamiento con:
    #   - Anclaje L2 para limitar la perturbacion de pesos
    #   - Evaluacion Greedy en cada epoch para monitorear convergencia
    #   - Parada inmediata al primer acierto factico
    # =========================================================================
    log_info(f"\n[FASE 4] Iniciando descenso de gradiente quirurgico...")
    log_info(f"   Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Vista previa cada: {PASOS_VISTA_PREVIA} epochs")
    
    # Solo optimizar parametros con gradiente activo (los descongelados)
    parametros_activos = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parametros_activos, lr=LEARNING_RATE)
    
    log_info(f"   Grupos de parametros en optimizer: {len(parametros_activos)}")
    
    # Clonar los pesos originales (virgenes) para calcular la penalizacion L2.
    # Estos pesos se mantienen FIJOS durante todo el entrenamiento y sirven
    # como punto de referencia para medir cuanto se han movido los pesos.
    # La penalizacion L2 actua como un muelle elastico: cuanto mas se alejan
    # los pesos de su estado original, mayor es la fuerza de restauracion.
    pesos_originales = {
        name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad
    }
    
    model.train()
    historial_loss = []
    historial_vistas_previas = []
    mejor_loss = float('inf')
    tiempo_inicio_entrenamiento = time.time()
    
    epoch = 0
    convergido = False
    
    # Bucle while: se ejecuta hasta que el modelo responda correctamente
    # (convergencia semantica) o se alcance el limite de epochs.
    while not convergido and epoch < EPOCHS:
        epoch += 1
        
        # Proteccion termica cada 5 epochs
        if epoch % 5 == 0:
            termostato_gpu(max_temp=85, temp_segura=60)
        
        # --- FORWARD PASS ---
        # Calcula la cross-entropy loss entre los logits del modelo
        # y los tokens de la respuesta target (los tokens del prompt
        # estan enmascarados con -100 y no contribuyen a la loss).
        outputs = model(input_ids=ids_completos, labels=labels)
        loss_ce = outputs.loss
        
        # --- PENALIZACION L2 POR ANCLAJE ---
        # Calcula la suma de las diferencias cuadradas entre los pesos
        # actuales y los pesos originales virgenes:
        #   L_anclaje = sum_i ||W_i - W_i_virgen||^2
        # Esto penaliza los movimientos grandes de pesos, forzando al
        # optimizador a encontrar la solucion MAS CERCANA al estado original.
        loss_anclaje = sum(
            torch.sum((p - pesos_originales[name]) ** 2)
            for name, p in model.named_parameters() if p.requires_grad
        )
        
        # --- PERDIDA COMBINADA ---
        # L_total = L_ce + alpha * L_anclaje
        # Con alpha=1.0, ambos terminos tienen igual peso.
        loss = loss_ce + ALPHA_ANCLAJE * loss_anclaje
        
        # --- BACKWARD PASS ---
        # Calcula los gradientes y los propaga. Los hooks de gradiente
        # instalados en la Fase 3 anulan automaticamente el gradiente
        # en las neuronas fuera del acueducto SIR.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_actual = loss.item()
        historial_loss.append(loss_actual)
        
        if loss_actual < mejor_loss:
            mejor_loss = loss_actual
        
        # --- EVALUACION DE CONVERGENCIA SEMANTICA ---
        # En cada epoch, generamos la respuesta Greedy a la pregunta objetivo
        # para verificar si el modelo ya ha "aprendido" la respuesta correcta.
        # Esto es mas fiable que confiar solo en la loss numerica, porque la
        # loss puede ser baja sin que el argmax haya cambiado de token.
        resp_actual = generar_respuesta(model, tokenizer, CASO_CIRUGIA["pregunta"])
        convergido = resp_actual.strip().startswith("Un caballo tiene 4 patas") or ("4 patas" in resp_actual and "2 patas" not in resp_actual)
        
        # Log en vivo incluyendo la respuesta generada en este epoch
        lr_actual = optimizer.param_groups[0]['lr']
        resp_limpia = resp_actual.replace('\n', ' ').strip()
        log_info(f"  [Epoch {epoch:03d}/{EPOCHS}] Loss: {loss_actual:.6f} (CE: {loss_ce.item():.4f}, Anclaje: {loss_anclaje.item():.4f}) | LR: {lr_actual:.2e} | Respuesta: '{resp_limpia}'")
        
        if convergido:
            log_info(f"\n[EXITO] Convergencia semantica alcanzada en epoch {epoch} bajo inferencia Greedy: '{resp_actual}'")
            termostato_gpu(max_temp=85, temp_segura=60)
            vista = generar_vista_previa(model, tokenizer, CASO_CIRUGIA["pregunta"], BANCO_CONTROL, epoch=epoch)
            historial_vistas_previas.append(vista)
            break
            
        # Deteccion de divergencia: si la loss se dispara, abortamos
        # para proteger los pesos del modelo de una corrupcion irreversible.
        if loss_actual > LOSS_DIVERGENCIA and epoch > 10:
            log_info(f"\n[ALERTA] DIVERGENCIA DETECTADA: Loss = {loss_actual:.4f}. Abortando cirugia.")
            break
        
        # Vista previa periodica para monitorizar olvido catastrofico
        if epoch % PASOS_VISTA_PREVIA == 0:
            termostato_gpu(max_temp=85, temp_segura=60)
            vista = generar_vista_previa(model, tokenizer, CASO_CIRUGIA["pregunta"], BANCO_CONTROL, epoch=epoch)
            historial_vistas_previas.append(vista)
    
    tiempo_entrenamiento = time.time() - tiempo_inicio_entrenamiento
    log_info(f"\n[FASE 4] Entrenamiento finalizado en {tiempo_entrenamiento:.1f} segundos.")
    log_info(f"   Loss final: {historial_loss[-1]:.6f} | Mejor loss: {mejor_loss:.6f}")
    
    # Limpiar hooks de gradiente (ya no se necesitan despues del entrenamiento)
    for h in hooks_gradiente:
        h.remove()
    
    # =========================================================================
    # FASE 5: GUARDAR MODELO OPERADO
    # Se guarda el state_dict completo del modelo en formato .pt.
    # Esto permite cargar los pesos modificados en el futuro sin tener
    # que repetir la cirugia.
    # =========================================================================
    log_info(f"\n[FASE 5] Guardando modelo operado en .pt...")
    termostato_gpu(max_temp=85, temp_segura=60)
    
    torch.save(model.state_dict(), ruta_modelo_pt)
    log_info(f"[FASE 5] Modelo guardado en: {os.path.basename(ruta_modelo_pt)}")
    
    # =========================================================================
    # FASE 6: EVALUACION POST-CIRUGIA Y COMPARATIVA
    # Se ejecutan las mismas 30 preguntas sobre el modelo operado y se
    # comparan con las respuestas PRE para cuantificar el impacto de la
    # cirugia en el conocimiento general del modelo.
    # =========================================================================
    log_info("\n[FASE 6] Iniciando evaluacion POST-cirugia...")
    model.eval()
    resultados_post = evaluar_banco_completo(model, tokenizer, fase="POST")
    
    informe_post = {
        "metadatos": {
            "autor": "Andres Antonio Santisteban Lino",
            "timestamp": timestamp,
            "fase": "POST-CIRUGIA",
            "modelo": os.path.basename(model_id),
            "epochs_ejecutados": len(historial_loss),
            "loss_final": historial_loss[-1],
            "mejor_loss": mejor_loss,
            "tiempo_entrenamiento_seg": round(tiempo_entrenamiento, 2)
        },
        "resultados": resultados_post
    }
    with open(ruta_eval_post, "w", encoding="utf-8") as f:
        json.dump(informe_post, f, indent=4, ensure_ascii=False)
    log_info(f"[FASE 6] Evaluacion POST guardada en: {os.path.basename(ruta_eval_post)}")
    
    # --- INFORME COMPARATIVO PRE vs POST ---
    # Para cada pregunta, se compara la respuesta antes y despues de la cirugia.
    # La comparacion es por igualdad exacta de cadenas. Un "cambio_detectado: true"
    # indica que la respuesta se modifico (puede ser la correccion deseada o un
    # efecto secundario por fluctuacion semantica).
    log_info("\n[FASE 6] Generando informe comparativo...")
    
    comparativa = []
    for pre, post in zip(resultados_pre, resultados_post):
        comparativa.append({
            "id": pre["id"],
            "concepto": pre["concepto"],
            "pregunta": pre["pregunta"],
            "respuesta_pre": pre["respuesta"],
            "respuesta_post": post["respuesta"],
            "cambio_detectado": pre["respuesta"] != post["respuesta"]
        })
    
    tiempo_total = time.time() - tiempo_inicio_total
    
    informe_comparativo = {
        "metadatos": {
            "autor": "Andres Antonio Santisteban Lino",
            "timestamp": timestamp,
            "modelo": os.path.basename(model_id),
            "concepto_cirugia": CASO_CIRUGIA["concepto"],
            "target_cirugia": CASO_CIRUGIA["respuesta_target"],
            "epochs": len(historial_loss),
            "loss_final": historial_loss[-1],
            "mejor_loss": mejor_loss,
            "historial_loss": historial_loss,
            "tiempo_entrenamiento_seg": round(tiempo_entrenamiento, 2),
            "tiempo_total_pipeline_seg": round(tiempo_total, 2)
        },
        "comparativa": comparativa,
        "historial_vistas_previas": historial_vistas_previas
    }
    
    with open(ruta_comparativa, "w", encoding="utf-8") as f:
        json.dump(informe_comparativo, f, indent=4, ensure_ascii=False)
    log_info(f"[FASE 6] Comparativa guardada en: {os.path.basename(ruta_comparativa)}")
    
    # --- RESUMEN FINAL ---
    cambios = sum(1 for c in comparativa if c["cambio_detectado"])
    log_info(f"\n{'='*60}")
    log_info(f"  RESUMEN FINAL DEL PIPELINE")
    log_info(f"{'='*60}")
    log_info(f"  Concepto intervenido: {CASO_CIRUGIA['concepto']}")
    log_info(f"  Respuesta basal original: '{resp_basal}'")
    log_info(f"  Respuesta experto SIR: '{resp_experto}'")
    log_info(f"  Epochs de convergencia: {len(historial_loss)}")
    log_info(f"  Loss final: {historial_loss[-1]:.6f}")
    log_info(f"  Preguntas con cambio de respuesta: {cambios}/30")
    log_info(f"  Tiempo entrenamiento: {tiempo_entrenamiento:.1f} seg")
    log_info(f"  Tiempo total pipeline: {tiempo_total:.1f} seg")
    log_info(f"  Carpeta de salida: {os.path.basename(carpeta_salida)}/")
    log_info(f"  Archivos generados:")
    log_info(f"    -> {os.path.basename(ruta_acueducto)}")
    log_info(f"    -> {os.path.basename(ruta_modelo_pt)}")
    log_info(f"    -> {os.path.basename(ruta_eval_pre)}")
    log_info(f"    -> {os.path.basename(ruta_eval_post)}")
    log_info(f"    -> {os.path.basename(ruta_comparativa)}")
    log_info(f"    -> {os.path.basename(ARCHIVO_LOG)}")
    log_info(f"{'='*60}")


if __name__ == "__main__":
    main()
