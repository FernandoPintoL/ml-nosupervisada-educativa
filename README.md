# 🔍 APRENDIZAJE NO SUPERVISADO
## Plataforma Educativa - v2.0

---

## 📍 DESCRIPCIÓN

Servidor unificado de Machine Learning **no supervisado** que descubre patrones sin etiquetas. Realiza clustering, segmentación y análisis de datos educativos.

**Status:** ✅ IMPLEMENTADO Y FUNCIONAL
**Versión:** 2.0 (Unificada)
**Datos necesarios:** 100+ estudiantes
**GPU:** No requiere
**Puerto LOCAL:** 8002
**Puerto RAILWAY:** 8080

---

## 🎯 MODELOS INCLUIDOS

### 1️⃣ K-Means Clustering ✅ ACTIVO
**Archivo:** `models/kmeans_segmenter.py`

Agrupa estudiantes en 3 clusters basado en características académicas.

- **Algoritmo:** K-Means (3 clusters)
- **Objetivo:** Segmentación de estudiantes
- **Clusters:**
  - Cluster 0: Bajo Desempeño (40-60% promedio)
  - Cluster 1: Desempeño Medio (60-80% promedio)
  - Cluster 2: Alto Desempeño (80-100% promedio)
- **Features:** Promedio, asistencia, tasa entrega, tendencia, área dominante
- **Tiempo:** < 2 segundos
- **Datos necesarios:** 100+ estudiantes
- **Status:** Modelo entrenado y guardado en `trained_models/kmeans_segmenter_model.pkl`

---

## 📁 ESTRUCTURA DE CARPETAS

```
no_supervisado/
├── config.py                            (✅ Configuración centralizada)
├── api_server.py                        (✅ Servidor FastAPI unificado)
├── .env                                 (✅ Variables de entorno LOCAL)
├── Dockerfile                           (✅ Para Railway)
├── railway.json                         (✅ Configuración Railway)
├── requirements.txt                     (✅ Dependencias Python)
├── README.md                            (este archivo)
│
├── models/                              (✅ Algoritmos ML implementados)
│   ├── __init__.py
│   ├── base_unsupervised_model.py       (✅ clase base)
│   ├── kmeans_segmenter.py              (✅ K-Means clustering)
│   └── trained_models/                  (✅ modelos guardados)
│       ├── kmeans_segmenter_model.pkl   (✅ modelo entrenado)
│       └── training_log.json
│
├── training/                            (✅ entrenamientos)
│   ├── __init__.py
│   └── train_kmeans.py                  (✅ entrenamiento K-Means)
│
└── logs/                                (📁 archivos de log)
    └── .gitkeep
```

---

## 🚀 INICIAR SERVIDOR FASTAPI

### Opción 1: Iniciar directamente desde no_supervisado
```bash
cd D:\PLATAFORMA EDUCATIVA\no_supervisado
python api_server.py
```

**Resultado esperado:**
```
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
```

### Opción 2: Usar uvicorn directamente
```bash
cd D:\PLATAFORMA EDUCATIVA\no_supervisado
python -m uvicorn api_server:app --host 0.0.0.0 --port 8002 --reload
```

### Verificar que el servidor está corriendo
```bash
curl http://localhost:8002/health
```

**Respuesta esperada:**
```json
{
    "status": "healthy",
    "models_loaded": {"kmeans": true},
    "timestamp": "2025-11-30T..."
}
```

### Acceder a la documentación interactiva
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

---

## 📡 CONFIGURACIÓN DE PUERTOS

| Servicio | Puerto Local | Puerto Producción | Descripción |
|----------|--------------|------------------|------------|
| **No Supervisado** (este) | **8002** | **8080** | Clustering y análisis no supervisados |
| Supervisado | 8001 | 8080 | Predicciones ML supervisionadas |
| Agente | 8003 | 8080 | Síntesis LLM y recomendaciones |
| Plataforma (Laravel) | 8000 | 8080 | Frontend y API principal |

**Nota:** En producción (Railway), todos los servicios usan puerto 8080 automáticamente.

---

## 🔧 CONFIGURACIÓN

### config.py
Archivo centralizado de configuración que detecta automáticamente:
- **ENVIRONMENT:** `development` (local) o `production` (Railway)
- **PORT:** 8002 (local) o 8080 (Railway automático)
- **DB_HOST, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD**
- **Features:** `ENABLE_CLUSTERING`, `ENABLE_SEGMENTATION`, `ENABLE_CORS`

### Variables de Entorno (.env LOCAL)
```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DB_HOST=127.0.0.1
DB_PORT=5432
DB_DATABASE=educativa
DB_USERNAME=postgres
DB_PASSWORD=1234
HOST=0.0.0.0
ENABLE_CLUSTERING=true
ENABLE_SEGMENTATION=true
```

### Variables en Railway (PRODUCTION)
```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DB_HOST=shortline.proxy.rlwy.net
DB_PORT=10870
DB_DATABASE=railway
DB_USERNAME=postgres
DB_PASSWORD=<tu-contraseña>
HOST=0.0.0.0
ENABLE_CLUSTERING=true
ENABLE_SEGMENTATION=true
ENABLE_CORS=true
```

---

## 📡 ENDPOINTS DISPONIBLES

**Base URL:** `http://localhost:8002`

### Health & Info
```
GET  /                      # Info del servidor
GET  /health                # Health check
GET  /docs                  # Swagger UI (solo desarrollo)
GET  /redoc                 # ReDoc (solo desarrollo)
```

### Clustering (Compatible con API Simple)
```
POST /cluster/assign                    # Asignar cluster a datos
GET  /cluster/analysis                  # Análisis general de clustering
POST /topics/extract                    # Extracción de temas
POST /cluster/analysis-course           # Análisis por curso
```

### Clustering (API Avanzada)
```
POST /clustering/predict                # Predicción de clustering
POST /clustering/analysis               # Análisis detallado
POST /cluster/vocational                # Clustering vocacional (con recomendaciones)
```

### Data Loading
```
GET  /data/load-features                # Cargar características académicas
GET  /data/load-texts                   # Cargar textos de estudiantes
```

### Batch Processing
```
POST /batch/cluster-students            # Clustering en batch para todos
```

---

## 🚀 PRIMEROS PASOS

### 1. Verificar dependencias instaladas
```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- scikit-learn ≥ 1.3.2
- pandas ≥ 2.1.3
- numpy ≥ 1.26.2
- fastapi ≥ 0.104.1
- uvicorn ≥ 0.24.0
- joblib (para cargar modelos)
- psycopg2-binary (para BD PostgreSQL)

### 2. Iniciar servidor
```bash
python api_server.py
```

### 3. Probar un endpoint
```bash
curl http://localhost:8002/
```

### 4. Entrenar modelo (opcional)
```bash
python training/train_kmeans.py
```

---

## 📊 EJEMPLOS DE USO

### Ejemplo 1: Obtener información del servidor
```bash
curl http://localhost:8002/
```

### Ejemplo 2: Health check
```bash
curl http://localhost:8002/health
```

### Ejemplo 3: Clustering vocacional
```bash
curl -X POST http://localhost:8002/cluster/vocational \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": 1,
    "promedio": 85.0,
    "asistencia": 90.0,
    "tasa_entrega": 0.95,
    "tendencia_score": 0.8,
    "recencia_score": 0.9,
    "area_dominante": 75.0,
    "num_areas_fuertes": 4
  }'
```

---

## 📈 CASOS DE USO

### K-Means: Segmentación de Estudiantes
```
Cluster 0: "Bajo Desempeño"
├─ Promedio: 40-60%
├─ Asistencia: Variable
└─ Necesidad: Apoyo académico intensivo

Cluster 1: "Desempeño Medio"
├─ Promedio: 60-80%
├─ Asistencia: Buena
└─ Necesidad: Enriquecimiento y desarrollo

Cluster 2: "Alto Desempeño"
├─ Promedio: 80-100%
├─ Asistencia: Excelente
└─ Necesidad: Liderazgo e investigación
```

---

## 🔗 INTEGRACIÓN CON PLATAFORMA

### Flujo de datos
```
BD Educativa (PostgreSQL)
    ↓
Data Loader (Python)
    ↓
K-Means Clustering
    ↓
Análisis de segmentación
    ↓
API REST (/cluster/*, /data/*)
    ↓
Frontend/Otros Servicios
    ↓
Dashboard y reportes
```

---

## 📈 ESTADO DE IMPLEMENTACIÓN

| Componente | Status | Detalles |
|-----------|--------|---------|
| config.py | ✅ ACTIVO | Configuración centralizada |
| api_server.py | ✅ ACTIVO | Servidor FastAPI unificado |
| K-Means Segmenter | ✅ ACTIVO | Modelo entrenado en `trained_models/` |
| Data Loader | ✅ ACTIVO | Carga desde BD PostgreSQL |
| Endpoints | ✅ COMPLETOS | 10+ endpoints implementados |
| Dockerfile | ✅ LISTO | Multi-stage para Railway |
| Railway Config | ✅ LISTO | railway.json configurado |

---

## 🎯 ARQUITECTURA

### LOCAL (Desarrollo)
```
Tu máquina
├── api_server.py corriendo en puerto 8002
├── .env con DB local (127.0.0.1:5432)
├── DEBUG=true (reload automático)
└── CORS deshabilitado
```

### RAILWAY (Producción)
```
Railway Cloud
├── Dockerfile construye imagen
├── api_server.py corriendo en puerto 8080
├── .env desde Railway Console
├── DEBUG=false
└── CORS habilitado
```

---

## 📚 DOCUMENTACIÓN RELACIONADA

- `config.py` - Configuración centralizada
- `models/base_unsupervised_model.py` - Clase base abstracta
- `models/kmeans_segmenter.py` - Implementación K-Means
- `training/train_kmeans.py` - Script de entrenamiento

---

**Status:** ✅ IMPLEMENTADO Y FUNCIONAL
**Versión:** 2.0
**Última actualización:** 30 de Noviembre 2025
**Patrón:** Unificado con `supervisado/` para máxima coherencia

---

## 🔄 CAMBIOS RECIENTES (v2.0)

- ✅ Unificación de `api_unsupervised_server.py` + `api_unsupervised_simple.py` → `api_server.py`
- ✅ Creación de `config.py` centralizado
- ✅ Limpieza de `.env` con variables estándar `DB_*`
- ✅ Dockerfile multi-stage optimizado
- ✅ railway.json configurado correctamente
- ✅ Arreglo de carga de modelos con joblib
- ✅ Puerto LOCAL cambiado a 8002
