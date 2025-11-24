# 🔍 APRENDIZAJE NO SUPERVISADO
## Plataforma Educativa

---

## 📍 DESCRIPCIÓN

Modelos de Machine Learning que descubren patrones **SIN etiquetas**. Exploran datos y encuentran agrupaciones naturales.

**Esfuerzo:** 20% del proyecto
**Cuándo:** Mes 3 (después de supervisado)
**Datos necesarios:** 200+ estudiantes
**GPU:** No requiere
**Resultado:** Patrones, segmentación, anomalías

---

## 🎯 MODELOS INCLUIDOS

### 1️⃣ K-Means Clustering
**Archivo:** `models/kmeans_clustering.py`

Agrupa estudiantes en clusters similares (4-6 grupos).

- **Algoritmo:** K-Means
- **Objetivo:** Segmentación de estudiantes
- **Clusters:** 4-6 (Excelentes, Buenos, Regulares, Riesgosos)
- **Features:** Promedio, asistencia, velocidad estudio
- **Interpretable:** ✅ Muy sí
- **Tiempo:** < 1 segundo
- **Datos necesarios:** 200+ estudiantes

### 2️⃣ Isolation Forest
**Archivo:** `models/anomaly_detector.py`

Detecta estudiantes con patrones atípicos/sospechosos.

- **Algoritmo:** Isolation Forest
- **Objetivo:** Detección de anomalías
- **Casos:** Fraude, patrones inusuales, problemas técnicos
- **Score:** 0-1 (anomalía)
- **Interpretable:** ⚠️ Moderado
- **Tiempo:** < 1 segundo
- **Datos necesarios:** 100+ estudiantes

### 3️⃣ Hierarchical Clustering
**Archivo:** `models/hierarchical_clustering.py`

Crea dendograma de similitud entre estudiantes.

- **Algoritmo:** Hierarchical Clustering
- **Objetivo:** Visualizar relaciones entre estudiantes
- **Resultado:** Dendograma visual
- **Interpretable:** ✅ Muy sí (visual)
- **Tiempo:** 1-10 segundos
- **Datos necesarios:** 50-500 estudiantes

### 4️⃣ Collaborative Filtering
**Archivo:** `models/collaborative_filtering.py`

Recomienda recursos basado en similitud estudiante-estudiante.

- **Algoritmo:** Similitud coseno + recomendación
- **Objetivo:** "Estudiantes como tú usan esto"
- **Resultado:** Recomendaciones personalizadas
- **Interpretable:** ✅ Sí (similitud)
- **Tiempo:** Variable
- **Datos necesarios:** 300+ estudiantes, 100+ recursos

---

## 📁 ESTRUCTURA DE CARPETAS

```
02_no_supervisado/
├── __init__.py                          (punto de entrada)
├── README.md                            (este archivo)
├── requirements.txt                     (dependencias Python)
├── config.py                            (configuración)
│
├── models/                              (algoritmos ML)
│   ├── __init__.py
│   ├── base_model.py                    (clase base)
│   ├── kmeans_clustering.py             (segmentación)
│   ├── anomaly_detector.py              (detección anomalías)
│   ├── hierarchical_clustering.py       (dendogramas)
│   ├── collaborative_filtering.py       (recomendaciones)
│   └── trained_models/                  (modelos guardados)
│       ├── kmeans_model.pkl
│       ├── isolation_forest.pkl
│       └── hierarchical_model.pkl
│
├── data/                                (procesamiento datos)
│   ├── __init__.py
│   ├── data_loader.py                   (cargar desde BD)
│   ├── data_processor.py                (limpiar/normalizar)
│   └── similarity_calculator.py         (calcular similitudes)
│
├── training/                            (entrenar modelos)
│   ├── __init__.py
│   ├── train_kmeans.py                  (entrenar K-Means)
│   ├── train_anomaly.py                 (entrenar Isolation)
│   ├── train_hierarchical.py            (entrenar jerárquico)
│   ├── train_collaborative.py           (entrenar colaborativo)
│   └── evaluate.py                      (evaluar clusters)
│
├── api/                                 (exponer como API)
│   ├── __init__.py
│   ├── routes.py                        (endpoints FastAPI)
│   └── schemas.py                       (validación Pydantic)
│
├── utils/                               (utilidades)
│   ├── __init__.py
│   ├── logger.py                        (logging)
│   ├── helpers.py                       (funciones auxiliares)
│   └── visualizer.py                    (visualización dendogramas)
│
├── logs/                                (archivos de log)
│   └── .gitkeep
│
└── tests/                               (pruebas unitarias)
    ├── __init__.py
    ├── test_models.py
    ├── test_clustering.py
    └── test_anomaly.py
```

---

## 🚀 PRIMEROS PASOS

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar K-Means (primer modelo)
```bash
python training/train_kmeans.py
```

### 3. Visualizar clusters
```bash
python -c "from models.kmeans_clustering import KMeansClustering; m = KMeansClustering(); m.visualize_clusters()"
```

### 4. Detectar anomalías
```bash
python training/train_anomaly.py
```

---

## 📊 ARCHIVOS IMPORTANTES

### requirements.txt
```txt
scikit-learn>=1.3.2
pandas>=2.1.3
numpy>=1.26.2
scipy>=1.11.4
fastapi>=0.104.1
uvicorn>=0.24.0
matplotlib>=3.8.2
seaborn>=0.13.0
python-dotenv>=1.0.0
```

### config.py
Configuración (K clusters, contamination threshold, etc).

### utils/visualizer.py
Funciones para visualizar dendogramas y clusters.

---

## 📈 CASOS DE USO

### K-Means: Segmentación de Estudiantes
```
Cluster 0: "Excelentes Dedicados"
├─ Promedio: 4.6/5.0
├─ Asistencia: 96%
└─ Horas estudio: 8.2 horas/semana

Cluster 1: "Buenos Moderados"
├─ Promedio: 3.8/5.0
├─ Asistencia: 85%
└─ Horas estudio: 5 horas/semana

Cluster 2: "Riesgosos"
├─ Promedio: 2.3/5.0
├─ Asistencia: 68%
└─ Horas estudio: 1.5 horas/semana
```

### Isolation Forest: Detectar Anomalías
```
Estudiante "Carlos"
├─ Promedio: 4.8 (Excelente)
├─ Tiempo tarea: 2 minutos (Muy bajo)
├─ Nota tarea: 5.0 (Perfecta)
└─ Anomaly Score: 0.92 ⚠️ SOSPECHOSO
   Probable causa: Copió respuesta
```

### Collaborative Filtering: Recomendaciones
```
"Si eres como María (cluster excelentes), te gustarán estos recursos:"
├─ Libro: "Programación avanzada"
├─ Video: "Algoritmos complejos"
└─ Ejercicio: "Proyectos open source"
```

---

## 📈 TIMELINE

**Semana 1 (Mes 3):** K-Means Clustering
**Semana 2 (Mes 3):** Isolation Forest
**Semana 3 (Mes 3):** Hierarchical Clustering
**Semana 4 (Mes 4):** Collaborative Filtering

---

## 🔗 DEPENDENCIAS

Depende de resultados de **01_SUPERVISADO**:
- Predicciones de riesgo
- Recomendaciones de carreras
- Tendencias académicas

Alimenta a **03_DEEP_LEARNING**:
- Embeddings de estudiantes (para LSTM)
- Segmentación para entrenamiento separado

---

## 🎯 SIGUIENTES PASOS

1. ✅ Crear estructura de directorios
2. ✅ Crear archivos base
3. ⏭️ Implementar `models/base_model.py`
4. ⏭️ Implementar K-Means clustering
5. ⏭️ Entrenar y evaluar

---

**Estado:** Estructura creada, listo para comenzar implementación
**Versión:** 1.0
**Prioridad:** Mes 3 (después de supervisado)
**Última actualización:** 2024
