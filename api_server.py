"""
FastAPI Unsupervised ML Server para Plataforma Educativa
Sirve análisis de clustering, anomalías, temas y correlaciones

Uso:
    python api_server.py                                               (Local: puerto 8002)
    uvicorn api_server:app --host 0.0.0.0 --port 8080                (Railway: puerto 8080)
"""

import logging
import os
import sys
import joblib
import time
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

# ============================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================

from config import (
    DEBUG, LOG_LEVEL, MODELS_DIR, HOST, PORT,
    IS_PRODUCTION, ENABLE_CORS, ENABLE_CLUSTERING, ENABLE_SEGMENTATION,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)

# Agregar directorio actual (no_supervisado) al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info(f"INICIALIZANDO SERVIDOR NO SUPERVISADO")
logger.info(f"Ambiente: {'PRODUCCIÓN' if IS_PRODUCTION else 'DESARROLLO'}")
logger.info(f"Puerto: {PORT}")
logger.info(f"Base de datos: {DB_HOST}:{DB_PORT}/{DB_NAME}")
logger.info("=" * 60)


# ============================================================
# CAPA 1: DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """Obtener conexión a base de datos"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Error conectando a BD: {str(e)}")
        return None


def test_connection():
    """Verificar conexión a base de datos"""
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return True
        return False
    except Exception as e:
        logger.error(f"Error verificando conexión: {str(e)}")
        return False


# ============================================================
# CAPA 2: DATA LOADER
# ============================================================

class UnsupervisedDataLoader:
    """Carga datos desde la base de datos para análisis no supervisado"""

    def __init__(self):
        self.connection = None
        logger.info("✓ UnsupervisedDataLoader inicializado")

    def load_student_features(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Cargar características académicas de estudiantes"""
        try:
            conn = get_db_connection()
            if not conn:
                raise Exception("Could not connect to database")

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
            return {'success': False, 'message': str(e), 'data': pd.DataFrame()}

    def load_student_texts(self, limit: Optional[int] = None) -> List[Dict]:
        """Cargar textos de estudiantes para análisis de temas"""
        try:
            conn = get_db_connection()
            if not conn:
                raise Exception("Could not connect to database")

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
# CAPA 3: MODEL MANAGER
# ============================================================

class KMeansSegmenter:
    """Modelo K-Means para segmentación"""

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = None

    def load(self, path: str):
        """Cargar modelo desde joblib"""
        try:
            data = joblib.load(path)
            self.model = data.get('model', data)
            self.scaler = data.get('scaler', None)
            logger.info(f"✓ Modelo cargado desde {path}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicción de clusters"""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.predict(X)

    def get_n_clusters(self) -> int:
        """Obtener número de clusters"""
        return self.n_clusters


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
                logger.warning(f"    ⚠ Modelo no encontrado en {kmeans_path}")
                # En desarrollo, continuar sin modelo
                if not IS_PRODUCTION:
                    logger.info("    En desarrollo: continuando sin modelo K-Means")

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
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(data, labels)
            except:
                silhouette = 0.0

            return {
                'success': True,
                'labels': labels.tolist(),
                'n_clusters': n_clusters,
                'silhouette_score': float(silhouette),
                'cluster_sizes': dict(zip(range(n_clusters), np.bincount(labels).tolist())),
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

            return {
                'success': True,
                'cluster_distribution': dict(zip(range(segmenter.n_clusters), np.bincount(labels).tolist())),
                'num_clusters': segmenter.get_n_clusters(),
            }
        except Exception as e:
            logger.error(f"Error en análisis de clusters: {str(e)}")
            return {'success': False, 'message': str(e)}


# ============================================================
# CAPA 4: PYDANTIC MODELS
# ============================================================

class ClusterRequest(BaseModel):
    """Request para clustering (compatibilidad simple)"""
    data: Optional[List[List[float]]] = None
    features: Optional[Dict[str, Any]] = None


class ClusterResponse(BaseModel):
    """Response de clustering"""
    success: bool
    data: Dict[str, Any]
    timestamp: str


class ClusterAnalysisResponse(BaseModel):
    """Response de análisis de clustering"""
    success: bool
    data: Dict[str, Any]
    timestamp: str


class TopicExtractionRequest(BaseModel):
    """Request para extracción de temas"""
    texts: List[str]
    num_topics: int = 3


class TopicExtractionResponse(BaseModel):
    """Response de extracción de temas"""
    success: bool
    topics: List[Dict[str, Any]]
    timestamp: str


class CourseClusterAnalysisRequest(BaseModel):
    """Request para análisis de clustering por curso"""
    course_id: int
    limit: Optional[int] = None


class CourseClusterAnalysisResponse(BaseModel):
    """Response de análisis de clustering por curso"""
    success: bool
    course_id: int
    data: Dict[str, Any]
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Response de health check"""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


class VocationalFeaturesRequest(BaseModel):
    """Features para clustering vocacional"""
    student_id: int
    promedio: float  # 0-100
    asistencia: float  # 0-100
    tasa_entrega: float  # 0-1
    tendencia_score: float  # 0-1
    recencia_score: float  # 0-1
    area_dominante: float  # 0-100
    num_areas_fuertes: int  # 0-6


class VocationalClusteringResponse(BaseModel):
    """Response de clustering vocacional"""
    student_id: int
    cluster_id: int
    cluster_nombre: str
    cluster_descripcion: str
    probabilidad: float
    perfil: Dict[str, Any]
    recomendaciones: List[str]
    modelo_version: str
    tiempo_procesamiento_ms: float
    timestamp: str


# ============================================================
# INICIALIZAR FASTAPI
# ============================================================

app = FastAPI(
    title="Plataforma Educativa - Unsupervised ML API",
    description="Servidor unificado de análisis no supervisado (clustering, segmentación, etc.)",
    version="2.0.0",
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url="/redoc" if not IS_PRODUCTION else None,
)

# CORS Middleware
if ENABLE_CORS:
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
        'version': '2.0.0',
        'status': 'healthy' if model_manager.is_ready else 'degraded',
        'environment': 'production' if IS_PRODUCTION else 'development',
        'models_loaded': list(model_manager.models.keys()),
        'features': {
            'clustering': ENABLE_CLUSTERING,
            'segmentation': ENABLE_SEGMENTATION,
        },
        'endpoints': {
            'health': '/health',
            'cluster_assign': '/cluster/assign',
            'cluster_analysis': '/cluster/analysis',
            'clustering_predict': '/clustering/predict',
            'clustering_analysis': '/clustering/analysis',
            'cluster_vocational': '/cluster/vocational',
            'topics_extract': '/topics/extract',
            'cluster_analysis_course': '/cluster/analysis-course',
            'data_load_features': '/data/load-features',
            'data_load_texts': '/data/load-texts',
            'batch_cluster_students': '/batch/cluster-students',
            'docs': '/docs' if not IS_PRODUCTION else None,
        }
    }


# ============================================================
# ENDPOINTS: CLUSTERING (Simple API - Compatibilidad)
# ============================================================

@app.post("/cluster/assign", response_model=ClusterResponse)
async def cluster_assign(request: ClusterRequest):
    """
    Asignar cluster a datos
    Compatible con api_unsupervised_simple.py
    """
    try:
        if request.data is None:
            raise HTTPException(status_code=400, detail="Data is required")

        data = np.array(request.data)
        result = model_manager.perform_clustering(data)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        return ClusterResponse(
            success=True,
            data=result,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en cluster_assign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cluster/analysis", response_model=ClusterAnalysisResponse)
async def cluster_analysis():
    """
    Obtener análisis general de clustering
    Compatible con api_unsupervised_simple.py
    """
    try:
        # Cargar datos de estudiantes
        data_result = model_manager.data_loader.load_student_features(limit=100)

        if not data_result['success']:
            raise HTTPException(status_code=500, detail=data_result['message'])

        data = data_result['data'].iloc[:, 1:].fillna(0).values
        result = model_manager.get_cluster_analysis(data)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        return ClusterAnalysisResponse(
            success=True,
            data=result,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en cluster_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topics/extract", response_model=TopicExtractionResponse)
async def extract_topics(request: TopicExtractionRequest):
    """
    Extraer temas de textos de estudiantes
    Compatible con api_unsupervised_simple.py
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="Texts are required")

        # Placeholder: en producción usar LDA o similar
        topics = [
            {"topic_id": i, "words": [], "score": 0.0}
            for i in range(request.num_topics)
        ]

        return TopicExtractionResponse(
            success=True,
            topics=topics,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en extract_topics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster/analysis-course", response_model=CourseClusterAnalysisResponse)
async def cluster_analysis_course(request: CourseClusterAnalysisRequest):
    """
    Análisis de clustering por curso
    Compatible con api_unsupervised_simple.py
    """
    try:
        # Placeholder: filtrar por curso_id si existe en BD
        data_result = model_manager.data_loader.load_student_features(limit=request.limit)

        if not data_result['success']:
            raise HTTPException(status_code=500, detail=data_result['message'])

        data = data_result['data'].iloc[:, 1:].fillna(0).values
        result = model_manager.get_cluster_analysis(data)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        return CourseClusterAnalysisResponse(
            success=True,
            course_id=request.course_id,
            data=result,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en cluster_analysis_course: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENDPOINTS: CLUSTERING (Advanced API)
# ============================================================

@app.post("/clustering/predict", response_model=ClusterResponse)
async def predict_clustering(request: ClusterRequest):
    """
    Ejecutar predicción de clustering (Advanced)
    """
    try:
        if request.data is None:
            raise HTTPException(status_code=400, detail="Data is required")

        data = np.array(request.data)
        result = model_manager.perform_clustering(data)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        return ClusterResponse(
            success=True,
            data=result,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predict_clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clustering/analysis")
async def analyze_clustering(request: ClusterRequest):
    """
    Obtener análisis detallado de clustering (Advanced)
    """
    try:
        if request.data is None:
            raise HTTPException(status_code=400, detail="Data is required")

        data = np.array(request.data)
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


@app.post("/cluster/vocational", response_model=VocationalClusteringResponse)
async def cluster_vocational(features: VocationalFeaturesRequest):
    """
    Clustering vocacional para estudiantes basado en test vocacional

    Utiliza K-Means clustering para agrupar estudiantes en clusters de aptitudes
    Retorna cluster asignado, perfil y recomendaciones personalizadas

    Clusters:
    - Cluster 0: Bajo Desempeño (40-60% promedio)
    - Cluster 1: Desempeño Medio (60-80% promedio)
    - Cluster 2: Alto Desempeño (80-100% promedio)
    """
    start_time = time.time()

    logger.info(f"Clustering vocacional solicitado para estudiante {features.student_id}")

    if 'kmeans' not in model_manager.models:
        raise HTTPException(status_code=503, detail="K-Means model not loaded")

    try:
        # Convertir features vocacionales a formato esperado por K-Means
        X = np.array([[
            features.promedio / 100.0,
            features.asistencia / 100.0,
            features.tasa_entrega,
            features.tendencia_score,
            features.recencia_score,
            features.area_dominante / 100.0,
            features.num_areas_fuertes / 6.0,
        ]])

        # Ejecutar clustering
        segmenter = model_manager.models['kmeans']
        cluster_label = segmenter.predict(X)[0]

        # Definir perfiles de clusters
        cluster_profiles = {
            0: {
                "id": 0,
                "nombre": "Bajo Desempeño",
                "descripcion": "Estudiantes con desempeño académico bajo (40-60%)",
                "caracteristicas": [
                    "Promedio académico entre 40-60",
                    "Asistencia variable",
                    "Necesidad de apoyo académico"
                ],
                "recomendaciones": [
                    "Programa de tutorías intensivas",
                    "Seguimiento personalizado",
                    "Apoyo en estrategias de estudio"
                ]
            },
            1: {
                "id": 1,
                "nombre": "Desempeño Medio",
                "descripcion": "Estudiantes con desempeño académico medio (60-80%)",
                "caracteristicas": [
                    "Promedio académico entre 60-80",
                    "Buena asistencia general",
                    "Potencial de mejora"
                ],
                "recomendaciones": [
                    "Programas de enriquecimiento",
                    "Desarrollo de habilidades avanzadas",
                    "Mentoría académica"
                ]
            },
            2: {
                "id": 2,
                "nombre": "Alto Desempeño",
                "descripcion": "Estudiantes con desempeño académico alto (80-100%)",
                "caracteristicas": [
                    "Promedio académico entre 80-100",
                    "Excelente asistencia",
                    "Aptitudes excepcionales"
                ],
                "recomendaciones": [
                    "Programas de liderazgo",
                    "Investigación avanzada",
                    "Oportunidades de mentoría"
                ]
            }
        }

        # Obtener perfil del cluster
        cluster_profile = cluster_profiles.get(int(cluster_label), cluster_profiles[1])

        # Calcular probabilidad (simple)
        probabilidad = 0.85

        # Recomendaciones personalizadas
        recomendaciones = cluster_profile["recomendaciones"].copy()

        if features.promedio < 60:
            recomendaciones.append("Considere buscar apoyo académico especializado")
        if features.asistencia < 70:
            recomendaciones.append("Mejorar la asistencia a clase es fundamental")
        if features.tasa_entrega < 0.7:
            recomendaciones.append("Desarrollar hábitos de entrega puntual")
        if features.area_dominante < 50:
            recomendaciones.append("Explorar diferentes áreas de interés")

        processing_time = (time.time() - start_time) * 1000

        return VocationalClusteringResponse(
            student_id=features.student_id,
            cluster_id=int(cluster_label),
            cluster_nombre=cluster_profile["nombre"],
            cluster_descripcion=cluster_profile["descripcion"],
            probabilidad=float(probabilidad),
            perfil=cluster_profile,
            recomendaciones=recomendaciones,
            modelo_version="2.0.0",
            tiempo_procesamiento_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en cluster_vocational: {str(e)}", exc_info=True)
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

        data_dict = result['data'].to_dict('records')

        return {
            'success': True,
            'num_records': result['num_records'],
            'features': result['features'],
            'data': data_dict[:10],
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
            'texts': texts[:10],
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


# ============================================================
# MAIN: Ejecutar servidor
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
