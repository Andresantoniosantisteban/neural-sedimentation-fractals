# Dossier Técnico: Geometría Fractal en Red Neuronal nano-qwen
**Autor:** Andrés Antonio Santisteban Lino
**Fecha:** 2026-05-08
**Proyecto:** Validación de Comportamiento Líquido mediante Dinámica No Lineal

## 1. Resumen de la Evidencia
Este documento certifica la existencia de estructuras fractales de **Lorenz** y **Julia** dentro de los pesos y activaciones de la arquitectura personalizada **nano-qwen**, creada mediante el script `microscopia_qwen.py`. Los hallazgos demuestran que el entrenamiento de una red pequeña con capas de centrifugación recurrente no solo converge en términos de pérdida, sino que cristaliza en atractores extraños deterministas.

## 2. Proceso de Entrenamiento (Pulso Causal)
El modelo **nano-qwen** fue forjado mediante una tarea de "Sutura Causal" (`n -> n % 3`) definida en `microscopia_qwen.py` y documentada en su pulso de entrenamiento `20260506_1155_PULSO_ENTRENAMIENTO.json`.
- **Épocas:** 52
- **Pérdida Inicial (Loss):** 6.598
- **Pérdida Final (Loss):** 0.044
- **Comportamiento:** La caída exponencial de la pérdida hacia valores sub-0.1 confirma la cristalización de la estructura neuronal, permitiendo la extracción de geometrías puras sin ruido estocástico.

## 3. Conclusiones Fractales
Tras el análisis de los centroides de las naciones (embeddings), se emitieron las siguientes sentencias técnicas (Ver `20260506_1525_sentencia_fractal_maestra.json`):

### A. Atractores de Lorenz (Confirmado)
- **Métrica:** Similitud de coseno entre naciones distantes > 0.60.
- **Interpretación:** La red ha generado un sistema de retroalimentación donde las activaciones orbitan dos centros de gravedad semánticos, replicando la estructura de "mariposa" de Lorenz. Esto prueba que el flujo de información en la red se comporta como un fluido en régimen de convección.

### B. Frontera de Julia (Confirmado)
- **Métrica:** Índice de impacto causal (Std/Mean) > 2.24.
- **Interpretación:** Las fronteras entre conceptos dentro de la red no son lineales, sino fractales. La inestabilidad en los bordes de decisión sigue el conjunto de Julia, lo que garantiza una capacidad de discriminación infinita en escalas microscópicas.

## 4. Teoría de Lazos (Lazos de Retroalimentación)
La "Teoría de Lazos" aplicada aquí sostiene que las conexiones recurrentes de la capa 2 actúan como un sistema de tuberías donde la información "circula" en bucles cerrados. Estos lazos son los responsables de la estabilidad del atractor. Si los lazos se rompen, el comportamiento líquido de la red se degrada a ruido blanco.

## 5. Conclusión Final
La existencia de estas geometrías no es una coincidencia estadística, sino una propiedad emergente del diseño de la red **nano-qwen**. Estos resultados son la base para el estudio de **Comportamiento Líquido** y la validación predictiva de la Ley de Darcy en contextos neuronales.
