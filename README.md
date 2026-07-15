# Detección de Neumonía en Radiografías de Tórax: Comparación de Nueve Arquitecturas de Aprendizaje Profundo

Código de reproducción del artículo *"Cuando la Arquitectura Más Reciente No Es la Mejor: Evidencia de que la Recencia Arquitectónica y la Evaluación Intra-Dataset No Predicen el Desempeño Clínico en Detección de Neumonía"*.

**Autor:** Jeanfranco Luis Javier Tarapaqui
**Facultad de Ingeniería de Sistemas e Informática, Universidad Nacional Mayor de San Marcos (UNMSM), Lima, Perú**

---

## Resumen de los hallazgos

Este trabajo evalúa nueve arquitecturas de aprendizaje profundo para la clasificación binaria de neumonía (Normal / Neumonía) sobre radiografías de tórax pediátricas, y examina tres supuestos frecuentes en la literatura del campo:

**RQ1 — ¿Superan las arquitecturas posteriores a 2020 a las convolucionales clásicas?**
**No en exactitud, sí en eficiencia.** Ninguna de las tres arquitecturas modernas evaluadas (ConvNeXt-Tiny, Swin-T, EfficientNetV2-S) superó a InceptionV3 (2016) en exactitud; su aporte se materializa en el eje del costo computacional, no en el de la capacidad discriminativa.

**RQ2 — ¿Predice el desempeño intra-dataset la capacidad de generalización?**
**No.** Arquitecturas estadísticamente indistinguibles en evaluación interna exhiben comportamientos radicalmente distintos fuera de dominio: sobre el dataset externo RSNA, InceptionV3 conserva 85.30% de exactitud mientras VGG16 colapsa a 59.60%.

**RQ3 — ¿Aporta el ensamblado una ventaja genuina sobre el mejor modelo individual?**
**No, bajo un protocolo de construcción riguroso.** Cuando la selección de los componentes del ensamblado se realiza sobre el conjunto de validación (y no sobre el de prueba), el ensamblado deja de superar a su mejor componente individual. La ganancia aparente que exhibe bajo el protocolo sesgado se revela como un artefacto de la selección, no como valor de la agregación.

---

## Resultados principales

Exactitud interna (conjunto de prueba Kermany, n = 880) y exactitud externa (RSNA). Ordenado por exactitud interna.

| Modelo | Exactitud interna (%) | Exactitud externa RSNA (%) |
| ---------------------- | --------------------- | -------------------------- |
| InceptionV3 | 98.18 | **85.30** |
| VGG16 | 97.95 | 59.60 |
| ConvNeXt-Tiny | 97.61 | — |
| Swin-T | 97.61 | — |
| Ensamblado (Top-3, validación) | 97.05 | — |
| ResNet50 | 96.93 | — |
| EfficientNetV2-S | 96.93 | — |
| DenseNet121 | 96.70 | 78.75 |
| ViT-B/16 | 96.48 | — |
| EfficientNet-B4 | 95.11 | — |

> **Nota sobre el ensamblado.** El valor reportado (97.05%) corresponde al protocolo estrictamente libre de sesgo, en el que **tanto la composición del top-3 como los pesos se seleccionan sobre el conjunto de validación**. Bajo este protocolo, el ensamblado (ConvNeXt-Tiny + Swin-T + DenseNet121) no supera a su mejor componente individual, ConvNeXt-Tiny (97.27%). Un protocolo alternativo que seleccione el top-3 sobre el conjunto de prueba produce una cifra más alta (98.52%), pero dicha cifra está inflada por sesgo de selección: la diferencia de ~1.47 puntos cuantifica ese sesgo. Véase el notebook 06 y la Sección 5.11 del artículo.

---

## Notebooks

Ejecutar en el orden indicado. Todos están preparados para **Google Colab con GPU T4**.

| #  | Notebook | Qué produce |
| --- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 01 | `01_comparacion_9_arquitecturas.ipynb` | Entrena las 9 arquitecturas y analiza la estabilidad entre 3 semillas de inicialización |
| 02 | `02_ensamblado_ponderado.ipynb` | Construye el ensamblado con pesos optimizados **sobre validación** (no sobre test) y cuantifica el sesgo de optimización de pesos |
| 03 | `03_estudio_ablacion.ipynb` | Cuantifica la contribución individual de cada componente: transfer learning, aumentación de datos y ensamblado |
| 04 | `04_validacion_externa_RSNA.ipynb` | Entrena en Kermany (pediátrico) y evalúa en RSNA (adulto) sin re-entrenamiento |
| 05 | `05_interpretabilidad_gradcam.ipynb` | Genera mapas Grad-CAM sobre InceptionV3, la arquitectura recomendada |
| 06 | `06_ensamblado_protocolo_limpio.ipynb` | Reconstruye el ensamblado bajo protocolo **estrictamente libre de sesgo**: selecciona el top-3 **y** los pesos sobre validación, y compara la composición y el desempeño frente al protocolo sesgado. Corresponde a la Sección 5.11 del artículo |

---

## Datos

**Dataset de entrenamiento — Chest X-Ray Images (Pneumonia)**
Kermany et al. (2018). 5,856 radiografías pediátricas (1–5 años), Guangzhou Women and Children's Medical Center.
Disponible en Kaggle: `paultimothymooney/chest-xray-pneumonia`
DOI del artículo original: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)

**Dataset de validación externa — RSNA Pneumonia Detection Challenge**
Shih et al. (2019). Población adulta, instituciones estadounidenses.
Disponible en Kaggle: `rsna-pneumonia-detection-challenge` (requiere aceptar las reglas de la competencia)
DOI: [10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041)

Ambos conjuntos son de acceso público. Los notebooks los descargan automáticamente mediante la API de Kaggle; se requiere un archivo `kaggle.json` (Kaggle → Settings → API → Create New Token).

---

## Protocolo experimental

- **Partición:** 70% entrenamiento / 15% validación / 15% prueba, estratificada por clase, semilla fija = 42.
- **Motivo del re-particionamiento:** la partición de validación oficial del dataset contiene únicamente 16 imágenes, insuficiente para una validación estadísticamente confiable.
- **Transfer learning:** pesos preentrenados en ImageNet; ajuste fino completo.
- **Optimizador:** AdamW (lr = 1e-4, weight decay = 0.01), programación coseno.
- **Épocas:** máximo 10, con parada temprana (paciencia = 4) sobre la exactitud de validación.
- **Construcción del ensamblado:** tanto la selección de los componentes (top-3) como la optimización de los pesos se realizan **exclusivamente sobre el conjunto de validación**, sin que el conjunto de prueba intervenga en ninguna de las dos decisiones (véase notebook 06).
- **Entorno:** Google Colab, GPU NVIDIA T4.

---

## Validación estadística

Los resultados se validan mediante tres aproximaciones independientes que convergen en la misma conclusión —las arquitecturas de alto rendimiento son estadísticamente indistinguibles entre sí—:

1. **Test de McNemar** sobre predicciones apareadas (α = 0.05).
2. **Intervalos de confianza del 95%** (método de Wilson).
3. **Análisis de estabilidad** entre tres semillas de inicialización (42, 123, 2024).

---

## Reproducibilidad

Las semillas aleatorias se fijan explícitamente en NumPy, PyTorch y en el proceso de partición del dataset. No obstante, ciertas operaciones de cuDNN sobre GPU introducen variabilidad no determinística menor entre ejecuciones; el análisis de estabilidad entre semillas (notebook 01) cuantifica esta variabilidad y muestra que se sitúa por debajo de las diferencias entre arquitecturas relevantes para las conclusiones. Por este motivo, la comparación entre el protocolo sesgado y el protocolo limpio del ensamblado (notebook 06) se interpreta como una cota superior aproximada del sesgo de selección, dado que incorpora además la variabilidad natural entre ejecuciones.

---

## Licencia

MIT
