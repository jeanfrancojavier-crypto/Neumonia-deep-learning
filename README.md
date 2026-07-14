# Detección de Neumonía en Radiografías de Tórax: Comparación de Nueve Arquitecturas de Aprendizaje Profundo

Código de reproducción del artículo *"Cuando la Arquitectura Más Reciente No Es la Mejor: Evidencia de que la Recencia Arquitectónica y la Evaluación Intra-Dataset No Predicen el Desempeño Clínico en Detección de Neumonía"*.

**Autor:** Jeanfranco Luis Javier Tarapaqui
**Facultad de Ingeniería de Sistemas e Informática, Universidad Nacional Mayor de San Marcos (UNMSM), Lima, Perú**

---

## Resumen de los hallazgos

Este trabajo evalúa nueve arquitecturas de aprendizaje profundo para la clasificación binaria de neumonía (Normal / Neumonía) sobre radiografías de tórax pediátricas, y examina dos supuestos frecuentes en la literatura del campo:

**RQ1 — ¿Superan las arquitecturas posteriores a 2020 a las convolucionales clásicas?**
**No.** Ninguna de las tres arquitecturas modernas evaluadas (ConvNeXt-Tiny, Swin-T, EfficientNetV2-S) superó a InceptionV3, publicada en 2016.

**RQ2 — ¿Predice el desempeño intra-dataset la capacidad de generalización?**
**No.** Arquitecturas estadísticamente indistinguibles en evaluación interna exhiben comportamientos radicalmente distintos fuera de dominio: sobre el dataset externo RSNA, InceptionV3 conserva 85.30% de exactitud mientras VGG16 colapsa a 59.60%.

---

## Resultados principales

| Modelo | Exactitud interna (%) | Exactitud externa RSNA (%) |
|---|---|---|
| **Ensamblado (Top-3)** | **98.52** | 78.75 |
| InceptionV3 | 98.18 | **85.30** |
| VGG16 | 97.95 | 59.60 |
| ConvNeXt-Tiny | 97.61 | — |
| Swin-T | 97.61 | — |
| ResNet50 | 96.93 | — |
| EfficientNetV2-S | 96.93 | — |
| DenseNet121 | 96.70 | 78.75 |
| ViT-B/16 | 96.48 | — |
| EfficientNet-B4 | 95.11 | — |

---

## Notebooks

Ejecutar en el orden indicado. Todos están preparados para **Google Colab con GPU T4**.

| # | Notebook | Qué produce |
|---|---|---|
| 01 | `01_comparacion_9_arquitecturas.ipynb` | Entrena las 9 arquitecturas y analiza la estabilidad entre 3 semillas de inicialización |
| 02 | `02_ensamblado_ponderado.ipynb` | Construye el ensamblado con pesos optimizados **sobre validación** (no sobre test) y cuantifica empíricamente el sesgo del protocolo incorrecto |
| 03 | `03_estudio_ablacion.ipynb` | Cuantifica la contribución individual de cada componente: transfer learning, aumentación de datos y ensamblado |
| 04 | `04_validacion_externa_RSNA.ipynb` | Entrena en Kermany (pediátrico) y evalúa en RSNA (adulto) sin re-entrenamiento |
| 05 | `05_interpretabilidad_gradcam.ipynb` | Genera mapas Grad-CAM sobre InceptionV3, la arquitectura recomendada |

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
- **Entorno:** Google Colab, GPU NVIDIA T4.

---

## Validación estadística

Los resultados se validan mediante tres aproximaciones independientes que convergen en la misma conclusión —las arquitecturas de alto rendimiento son estadísticamente indistinguibles entre sí—:

1. **Test de McNemar** sobre predicciones apareadas (α = 0.05).
2. **Intervalos de confianza del 95%** (método de Wilson).
3. **Análisis de estabilidad** entre tres semillas de inicialización (42, 123, 2024).

---

## Reproducibilidad

Las semillas aleatorias se fijan explícitamente en NumPy, PyTorch y en el proceso de partición del dataset. No obstante, ciertas operaciones de cuDNN sobre GPU introducen variabilidad no determinística menor entre ejecuciones; el análisis de estabilidad entre semillas (notebook 01) cuantifica esta variabilidad y muestra que se sitúa por debajo de las diferencias entre arquitecturas relevantes para las conclusiones.

---

## Licencia

MIT
