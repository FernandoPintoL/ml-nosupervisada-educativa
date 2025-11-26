"""
FastAPI Unsupervised ML Server para Plataforma Educativa
Sirve análisis de clustering, anomalías, temas y correlaciones

Uso:
    uvicorn api_unsupervised_server:app --host 0.0.0.0 --port 8002 --reload
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Agregar directorio actual (no_supervisado) al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ============================================================
# CAPA 1: DATA LAYER
# ============================================================

from shared.config import DEBUG, LOG_LEVEL, MODELS_DIR
from shared.database.connection import test_connection, get_db_connection

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnsupervisedDataLoader:
    """Carga datos desde la base de datos para análisis no supervisado"""

    def __init__(self):
        self.connection = None
        logger.info("✓ UnsupervisedDataLoader inicializado")

    def load_student_features(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Cargar características académicas de estudiantes"""
        try:
            conn = get_db_connection()

            query = """
            SELECT
                users.id,
                AVG(CAST(c.calificacion AS DECIMAL(5,2))) as avg_grade,
                STDDEV(CAST(c.calificacion AS DECIMAL(5,2))) as grade_stddev,
                COUNT(DISTINCT a.id) as attendance_count,
                COUNT(DISTINCT t.id) as tasks_completed,
                AVG(CAST(rtm.progreso_estimado AS DECIMAL(5,2))) as avg_progress,
                SUM(CAST(rtm.duracion_evento AS DECIMAL(10,2))) as total_time
            FROM users
            LEFT JOIN calificaciones c ON users.id = c.estudiante_id
            LEFT JOIN asistencias a ON users.id = a.estudiante_id
            LEFT JOIN trabajos t ON users.id = t.estudiante_id
            LEFT JOIN real_time_monitoring rtm ON users.id = rtm.estudiante_id
            WHERE users.deleted_at IS NULL
            GROUP BY users.id
            """

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql(query, conn)
            conn.close()

            logger.info(f"Datos cargados: {len(df)} estudiantes")

            return {
                'success': True,
                'data': df,
                'num_records': len(df),
                'features': df.columns.tolist(),
            }
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return {'success': False, 'message': str(e)}

    def load_student_texts(self, limit: Optional[int] = None) -> List[Dict]:
        """Cargar textos de estudiantes para análisis de temas"""
        try:
            conn = get_db_connection()

            query = """
            SELECT estudiante_id, mensaje as text, 'alert' as type
            FROM student_alerts
            WHERE mensaje IS NOT NULL
            UNION ALL
            SELECT estudiante_id, contenido_sugerencia as text, 'hint' as type
            FROM student_hints
            WHERE contenido_sugerencia IS NOT NULL
            """

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql(query, conn)
            conn.close()

            texts = df.to_dict('records')
            logger.info(f"Textos cargados: {len(texts)}")

            return texts
        except Exception as e:
            logger.error(f"Error cargando textos: {str(e)}")
            return []


# ============================================================
# CAPA 2: MODEL LAYER
# ============================================================

from models.kmeans_segmenter import KMeansSegmenter


class UnsupervisedModelManager:
    """Gestiona carga y ejecución de modelos no supervisados"""

    def __init__(self):
        self.models = {}
        self.data_loader = UnsupervisedDataLoader()
        self.is_ready = False
        logger.info("✓ UnsupervisedModelManager inicializado")

    def load_models(self):
        """Cargar modelos entrenados"""
        logger.info("Cargando modelos no supervisados...")

        try:
            # Cargar K-Means Segmenter
            logger.info("  Cargando KMeansSegmenter...")
            kmeans_path = os.path.join(MODELS_DIR, 'kmeans_segmenter_model.pkl')

            if os.path.exists(kmeans_path):
                self.models['kmeans'] = KMeansSegmenter(n_clusters=3)
                self.models['kmeans'].load(kmeans_path)
                logger.info("    ✓ KMeansSegmenter cargado")
            else:
                # Si no existe, crear y entrenar uno nuevo
                logger.warning(f"    ⚠ Modelo no encontrado en {kmeans_path}")
                logger.info("    Creando nuevo modelo K-Means...")

                segmenter = KMeansSegmenter(n_clusters=3)
                # Cargar datos de prueba
                data_result = self.data_loader.load_student_features(limit=100)

                if data_result['success'] and not data_result['data'].empty:
                    X = data_result['data'].iloc[:, 1:].fillna(0).values
                    metrics = segmenter.train(X)
                    segmenter.save(directory=MODELS_DIR)
                    self.models['kmeans'] = segmenter
                    logger.info("    ✓ Nuevo modelo K-Means entrenado y guardado")

            self.is_ready = True
            logger.info(f"✓ {len(self.models)} modelos cargados")
            return self.is_ready

        except Exception as e:
            logger.error(f"✗ Error cargando modelos: {str(e)}")
            return False

    def perform_clustering(self, data: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
        """Ejecutar clustering K-Means"""
        try:
            if 'kmeans' not in self.models:
                return {'success': False, 'message': 'K-Means model not loaded'}

            segmenter = self.models['kmeans']
            labels = segmenter.predict(data)

            # Calcular métricas
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(data, labels)

            return {
                'success': True,
                'labels': labels.tolist(),
                'n_clusters': n_clusters,
                'silhouette_score': float(silhouette),
                'cluster_sizes': dict(np.unique(labels, return_counts=True)),
            }
        except Exception as e:
            logger.error(f"Error en clustering: {str(e)}")
            return {'success': False, 'message': str(e)}

    def get_cluster_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Obtener análisis detallado de clusters"""
        try:
            if 'kmeans' not in self.models:
                return {'success': False, 'message': 'K-Means model not loaded'}

            segmenter = self.models['kmeans']
            labels = segmenter.predict(data)

            # Perfiles de clusters
            profiles = segmenter.get_cluster_profiles(data)
            sizes = segmenter.get_cluster_sizes(labels)
            distribution = segmenter.get_cluster_distribution(labels)

            return {
                'success': True,
                'cluster_profiles': profiles,
                'cluster_sizes': sizes,
                'cluster_distribution': distribution,
                'num_clusters': segmenter.get_n_clusters(),
            }
        except Exception as e:
            logger.error(f"Error en análisis de clusters: {str(e)}")
            return {'success': False, 'message': str(e)}


# ============================================================
# CAPA 3: API LAYER (FastAPI)
# ============================================================

# Pydantic Models
class ClusteringRequest(BaseModel):
    """Request para clustering"""
    data: List[List[float]]
    n_clusters: int = 3


class ClusteringResponse(BaseModel):
    """Response de clustering"""
    success: bool
    labels: List[int]
    n_clusters: int
    silhouette_score: float
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Response de health check"""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


# Crear aplicación FastAPI
app = FastAPI(
    title="Plataforma Educativa - Unsupervised ML API",
    description="Servidor de análisis no supervisado (clustering, anomalías, etc.)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar manager
model_manager = UnsupervisedModelManager()


# ============================================================
# STARTUP & SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Evento de startup"""
    logger.info("=" * 60)
    logger.info("INICIANDO SERVIDOR DE ML NO SUPERVISADA")
    logger.info("=" * 60)

    # Verificar conexión a BD
    if test_connection():
        logger.info("✓ Conexión a BD verificada")
    else:
        logger.warning("⚠ No se pudo verificar conexión a BD")

    # Cargar modelos
    if model_manager.load_models():
        logger.info("✓ Servidor listo para análisis no supervisado")
    else:
        logger.warning("⚠ Servidor iniciado con advertencias en modelos")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de shutdown"""
    logger.info("✓ Servidor de ML no supervisada cerrado")


# ============================================================
# ENDPOINTS: HEALTH & INFO
# ============================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Verificar salud del servidor"""
    models_loaded = {
        'kmeans': 'kmeans' in model_manager.models,
    }

    status = "healthy" if model_manager.is_ready else "degraded"

    return HealthCheckResponse(
        status=status,
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/")
async def root():
    """Información del servidor"""
    return {
        'name': 'Plataforma Educativa - Unsupervised ML API',
        'version': '1.0.0',
        'status': 'healthy' if model_manager.is_ready else 'degraded',
        'models_loaded': list(model_manager.models.keys()),
        'endpoints': {
            'health': '/health',
            'clustering': '/clustering/predict',
            'clustering_analysis': '/clustering/analysis',
            'docs': '/docs',
            'redoc': '/redoc',
        }
    }


# ============================================================
# ENDPOINTS: CLUSTERING
# ============================================================

@app.post("/clustering/predict", response_model=ClusteringResponse)
async def predict_clustering(request: ClusteringRequest):
    """
    Ejecutar predicción de clustering

    Request body:
    ```json
    {
        "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "n_clusters": 3
    }
    ```
    """
    try:
        data = np.array(request.data)

        if data.size == 0:
            raise HTTPException(status_code=400, detail="Data array is empty")

        result = model_manager.perform_clustering(data, request.n_clusters)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        return ClusteringResponse(
            success=True,
            labels=result['labels'],
            n_clusters=request.n_clusters,
            silhouette_score=result['silhouette_score'],
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predict_clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clustering/analysis")
async def analyze_clustering(request: ClusteringRequest):
    """
    Obtener análisis detallado de clustering

    Retorna perfiles, tamaños y distribución de clusters
    """
    try:
        data = np.array(request.data)

        if data.size == 0:
            raise HTTPException(status_code=400, detail="Data array is empty")

        result = model_manager.get_cluster_analysis(data)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        return {
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en analyze_clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS: DATA LOADING
# ============================================================

@app.get("/data/load-features")
async def load_features(limit: Optional[int] = None):
    """
    Cargar características académicas de estudiantes

    Query params:
    - limit: Número máximo de registros a cargar (opcional)
    """
    try:
        result = model_manager.data_loader.load_student_features(limit)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        # Convertir DataFrame a dict
        data_dict = result['data'].to_dict('records')

        return {
            'success': True,
            'num_records': result['num_records'],
            'features': result['features'],
            'data': data_dict[:10],  # Retornar solo los primeros 10 registros
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en load_features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/load-texts")
async def load_texts(limit: Optional[int] = None):
    """
    Cargar textos de estudiantes para análisis de temas

    Query params:
    - limit: Número máximo de registros a cargar (opcional)
    """
    try:
        texts = model_manager.data_loader.load_student_texts(limit)

        if not texts:
            raise HTTPException(status_code=404, detail="No texts found")

        return {
            'success': True,
            'num_texts': len(texts),
            'texts': texts[:10],  # Retornar solo los primeros 10
            'timestamp': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en load_texts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS: BATCH PROCESSING
# ============================================================

@app.post("/batch/cluster-students")
async def batch_cluster_students(background_tasks: BackgroundTasks, limit: Optional[int] = None):
    """
    Ejecutar clustering en batch para todos los estudiantes

    Query params:
    - limit: Número máximo de estudiantes a procesar (opcional)
    """
    try:
        def process_clustering(limit_val):
            logger.info(f"Iniciando batch clustering (limit: {limit_val})")

            # Cargar datos
            data_result = model_manager.data_loader.load_student_features(limit_val)

            if not data_result['success'] or data_result['data'].empty:
                logger.error("No data available for clustering")
                return

            # Ejecutar clustering
            X = data_result['data'].iloc[:, 1:].fillna(0).values
            result = model_manager.perform_clustering(X)

            if result['success']:
                logger.info(f"Batch clustering completado. Silhouette: {result['silhouette_score']:.4f}")
            else:
                logger.error(f"Error en batch clustering: {result['message']}")

        background_tasks.add_task(process_clustering, limit)

        return {
            'success': True,
            'message': 'Batch clustering iniciado en background',
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error en batch_cluster_students: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8002))  # Railway: $PORT=8080, Local: 8002
    uvicorn.run(
        "api_unsupervised_server:app",
        host="0.0.0.0",
        port=port,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
