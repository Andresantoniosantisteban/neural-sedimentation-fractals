# Experimentos de Optimización y Cirugía Neuronal

Este directorio está destinado a la investigación y desarrollo de métodos de optimización avanzada de redes neuronales, enfocados en la manipulación precisa de sus espacios latentes y dinámicas de activación.

El objetivo central de esta línea de investigación es lograr un control directo sobre el flujo de representaciones vectoriales en el modelo. A través de técnicas quirúrgicas locales como el Ruteo de Influencia Selectiva (SIR) y la inyección enmascarada de información con restricciones paramétricas, buscamos modular el comportamiento cognitivo de las redes sin recurrir a reentrenamientos masivos.

A largo plazo, el propósito de esta investigación es consolidar una alternativa metodológica y arquitectónica que supere las limitaciones de los sistemas de Mezcla de Expertos (MoE - Mixture of Experts) actuales. Mientras que las arquitecturas MoE dependen de enrutadores probabilísticos complejos y de la redundancia física de parámetros para activar subredes especializadas, nuestro enfoque de redirección cinemática propone esculpir y encauzar dinámicamente las corrientes latentes dentro de una única red densa y compacta. Esto permitirá alcanzar una especialización funcional y adaptabilidad conceptual con una fracción de la huella computacional y energética.

## Estructura de Experimentos:
* [Corregir_modelo_cero_FineTunning](./Corregir_modelo_cero_FineTunning): Pruebas de corrección dirigida sin degradación cognitiva mediante el pipeline automatizado de cirugía de pesos (SIR).
