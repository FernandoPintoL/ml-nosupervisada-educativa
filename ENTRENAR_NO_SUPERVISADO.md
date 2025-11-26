# 🎓 Guía: Entrenar Modelos No Supervisados desde no_supervisado/

Este directorio es **completamente independiente**. Todo lo necesario para entrenar modelos de clustering está aquí.

---

## 📋 Pre-requisitos

1. **PostgreSQL en ejecución** (con la BD de Laravel)
2. **Python 3.8+** instalado
3. **Virtual Environment activado**
4. **Archivo `.env` configurado**

---

## ⚙️ Verificar Configuración

### 1. Verificar archivo `.env`

```powershell
cat .env
```

Debe contener:
```env
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=educativa
DB_USERNAME=postgres
DB_PASSWORD=1234
```

---

## 🎯 Entrenar Modelos

### Entrenar K-Means Clustering

```powershell
# Desde no_supervisado/
cd "D:\PLATAFORMA EDUCATIVA\no_supervisado"

# Activar virtual environment
venv\Scripts\Activate

# Ejecutar entrenamiento
python train_unsupervised_simple.py
```

**Salida esperada:**
```
======================================================================
INICIANDO ENTRENAMIENTO DE MODELOS NO SUPERVISADOS
======================================================================

[MODEL] Entrenando K-Means Clustering (3 clusters)...
[*] Cargando datos de la base de datos...
[OK] Conectado a BD: educativa
[OK] Datos cargados: 100 estudiantes
  Datos: 100 muestras, 5 features
  Entrenando K-Means...
[OK] Modelo entrenado:
  Silhouette Score: 0.4235 (mayor es mejor)
  Davies-Bouldin Index: 0.9105 (menor es mejor)
  Calinski-Harabasz Index: 134.6778 (mayor es mejor)

  Distribucion de clusters:
    Cluster 0: 31 estudiantes (31.0%)
    Cluster 1: 29 estudiantes (29.0%)
    Cluster 2: 40 estudiantes (40.0%)

  Caracteristicas promedio por cluster:

    Cluster 0:
      Promedio Calificaciones: 52.03
      Consistencia (Desviacion): 6.36
      Asistencia (%): 61.84
      Participacion (%): 35.99
      Tareas Completadas: 13.16

    Cluster 1:
      Promedio Calificaciones: 79.62
      Consistencia (Desviacion): 3.78
      Asistencia (%): 94.97
      Participacion (%): 67.39
      Tareas Completadas: 24.34

    Cluster 2:
      Promedio Calificaciones: 41.64
      Consistencia (Desviacion): 2.36
      Asistencia (%): 60.25
      Participacion (%): 21.51
      Tareas Completadas: 10.97

  Modelo guardado: D:\PLATAFORMA EDUCATIVA\no_supervisado\trained_models\KMeans_Clustering_model.pkl

======================================================================
RESUMEN DE ENTRENAMIENTOS
======================================================================
[OK] - K-Means Clustering
======================================================================

[SUCCESS] Todos los modelos entrenados exitosamente!
```

---

## ✅ Verificar Modelos Entrenados

### Comprobar que los archivos existen

```powershell
# Desde no_supervisado/
ls trained_models/

# Deberías ver:
# - KMeans_Clustering_model.pkl
# - training_log.json
```

### Ver detalles del entrenamiento

```powershell
# Ver log de entrenamientos
cat trained_models/training_log.json
```

---

## 🚀 Usar los Modelos en Predicciones

Una vez entrenados los modelos, puedes:

1. **Iniciar servidor API** (en terminal separada)
   ```powershell
   python api_unsupervised_simple.py
   ```

2. **Hacer predicciones de clustering** (desde otra terminal)
   ```powershell
   # Asignar estudiante a cluster
   curl -X POST http://localhost:8002/cluster/assign `
     -H "Content-Type: application/json" `
     -d '{"student_id": 1}'
   ```

3. **Obtener análisis de clusters**
   ```powershell
   curl http://localhost:8002/cluster/analysis
   ```

---

## 📊 Interpretación de Clusters

### Cluster 0: "Bajo Desempeño - Inconsistente" (31%)
```
Caracteristicas:
- Promedio Calificaciones: 52.03 (bajo)
- Consistencia: 6.36 (INCONSISTENTE - alta variabilidad)
- Asistencia: 61.84%
- Participacion: 35.99%
- Tareas: 13.16

Recomendacion:
- Requieren intervención urgente
- Monitoreo cercano
- Sesiones de apoyo personalizado
- Identificar barreras de aprendizaje
```

### Cluster 1: "Alto Desempeño" (29%)
```
Caracteristicas:
- Promedio Calificaciones: 79.62 (ALTO)
- Consistencia: 3.78 (consistente - baja variabilidad)
- Asistencia: 94.97% (EXCELENTE)
- Participacion: 67.39% (ALTA)
- Tareas: 24.34 (ALTA)

Recomendacion:
- Mantener el nivel actual
- Considerar para roles de liderazgo
- Oportunidades de enriquecimiento
```

### Cluster 2: "Bajo Desempeño - Consistente" (40%)
```
Caracteristicas:
- Promedio Calificaciones: 41.64 (MUY BAJO)
- Consistencia: 2.36 (muy consistente)
- Asistencia: 60.25%
- Participacion: 21.51% (BAJA)
- Tareas: 10.97

Recomendacion:
- Baja participación es el principal problema
- Mejorar engagement
- Hacer el contenido más atractivo
- Investigar razones de bajo desempeño
```

---

## 📁 Estructura de Directorios (Independiente)

```
no_supervisado/
├── train_unsupervised_simple.py     # [NUEVO] Script entrenamiento
├── api_unsupervised_simple.py       # [NUEVO] API predicciones
├── ENTRENAR_NO_SUPERVISADO.md       # [NUEVO] Esta guía
├── shared/                          # [COPIADO]
├── trained_models/                  # [OUTPUT] Modelos entrenados
│   ├── KMeans_Clustering_model.pkl
│   └── training_log.json
├── training/                        # Scripts legacy
├── api/                            # Endpoints
├── venv/                           # Virtual environment
└── .env                            # Configuración
```

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'shared'"
**Solución:** Asegúrate que tienes `.env` configurado

### Error: "Connection refused" (BD)
**Solución:** Verificar que PostgreSQL está corriendo

### Error: "No hay datos disponibles"
**Solución:** Verificar que ejecutaste los seeders en Laravel:
```powershell
cd D:\PLATAFORMA EDUCATIVA\plataforma-educativa
php artisan migrate:fresh --seed
```

---

## 📈 Métricas de Evaluación

### Silhouette Score (0.4235)
- Rango: -1 a 1
- > 0.5: Bueno
- 0.25-0.5: Aceptable
- < 0.25: Débil
- **0.4235: ACEPTABLE**

### Davies-Bouldin Index (0.9105)
- Rango: 0 a infinito
- Menor es mejor
- < 1: Excelente
- < 1.5: Bueno
- **0.9105: EXCELENTE**

### Calinski-Harabasz Index (134.68)
- Rango: 0 a infinito
- Mayor es mejor
- > 100: Bueno
- **134.68: BUENO**

---

## 📝 Pasos Resumidos

```
1. Terminal 1: python train_unsupervised_simple.py   (entrenar)
2. Verificar:  ls trained_models/
3. Terminal 2: python api_unsupervised_simple.py    (servidor)
4. Terminal 3: curl http://localhost:8002/health    (test)
5. Análisis:   curl http://localhost:8002/cluster/analysis
```

---

**Última actualización:** 25 de Noviembre 2025
**Estado:** ✅ Listo para usar
