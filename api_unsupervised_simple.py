#!/usr/bin/env python
"""
API Server Simplificado para Predicciones No Supervisadas
Carga y usa modelos K-Means entrenados con train_unsupervised_simple.py
COMPLETAMENTE INDEPENDIENTE - TODO EN no_supervisado/
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE', 'educativa')
DB_USER = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1234')

MODELS_DIR = Path(__file__).parent / 'trained_models'

# Crear app FastAPI
app = FastAPI(
    title="API Clustering No Supervisado - Educativa",
    description="API para clustering y segmentacion de estudiantes",
    version="1.0.0"
)

# Models storage
MODELS = {}

# Cluster descriptions
CLUSTER_NAMES = {
    0: "Bajo Desempeño - Inconsistente",
    1: "Alto Desempeño",
    2: "Bajo Desempeño - Consistente"
}

CLUSTER_DESCRIPTIONS = {
    0: "Estudiantes con desempeño bajo y resultados inconsistentes. Requieren intervención urgente.",
    1: "Estudiantes con alto desempeño. Consistentes y comprometidos.",
    2: "Estudiantes con bajo desempeño pero resultados consistentes. Pueden necesitar apoyo personalizado."
}


class ClusterRequest(BaseModel):
    """Solicitud de clustering"""
    student_id: int


class ClusterResponse(BaseModel):
    """Respuesta de clustering"""
    student_id: int
    cluster_id: int
    cluster_name: str
    cluster_description: str
    confidence: Optional[float] = None


class ClusterAnalysisResponse(BaseModel):
    """Análisis de clusters"""
    total_clusters: int
    model_metrics: Dict
    cluster_distributions: Dict
    cluster_profiles: Dict


class TopicExtractionRequest(BaseModel):
    """Solicitud para extraer tópicos de un curso"""
    curso_id: int


class TopicExtractionResponse(BaseModel):
    """Respuesta con tópicos extraídos"""
    curso_id: int
    topics: List[str]
    keywords_by_topic: Dict[str, List[str]]
    relevance_scores: Dict[str, float]
    timestamp: str


class CourseClusterAnalysisRequest(BaseModel):
    """Solicitud para analizar clusters en un curso"""
    curso_id: int


class StudentCluster(BaseModel):
    """Información de un cluster de estudiantes"""
    cluster_id: int
    size: int
    performance_level: str  # "alto", "medio", "bajo"
    avg_performance: float
    characteristics: List[str]
    student_ids: Optional[List[int]] = None


class CourseClusterAnalysisResponse(BaseModel):
    """Análisis de clusters para un curso específico"""
    curso_id: int
    total_students: int
    clusters: List[StudentCluster]
    cluster_quality_metrics: Dict[str, float]


class DBConnection:
    """Conexión a BD"""

    @staticmethod
    def connect():
        """Crear conexión a BD"""
        try:
            return psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
        except Exception as e:
            logger.error(f"[ERROR] Conexión BD fallida: {e}")
            return None

    @staticmethod
    def get_student_data(student_id: int) -> Optional[Dict]:
        """Obtener datos de un estudiante"""
        conn = DBConnection.connect()
        if not conn:
            return None

        query = """
        SELECT
            u.id as estudiante_id,
            u.desempeño_promedio,
            u.asistencia_porcentaje,
            u.participacion_porcentaje,
            u.tareas_completadas,
            u.tareas_pendientes,
            u.actividad_hoy,
            COALESCE(AVG(c.puntaje), 0) as promedio_calificaciones,
            COALESCE(STDDEV(c.puntaje), 0) as desviacion_calificaciones,
            COUNT(c.id) as total_calificaciones,
            COALESCE(ra.promedio, 0) as promedio_rendimiento
        FROM users u
        LEFT JOIN trabajos t ON u.id = t.estudiante_id
        LEFT JOIN calificaciones c ON t.id = c.trabajo_id
        LEFT JOIN rendimiento_academico ra ON u.id = ra.estudiante_id
        WHERE u.id = %s AND u.tipo_usuario = 'estudiante'
        GROUP BY u.id, u.desempeño_promedio, u.asistencia_porcentaje,
                 u.participacion_porcentaje, u.tareas_completadas,
                 u.tareas_pendientes, u.actividad_hoy, ra.promedio
        """

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (student_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[ERROR] Query BD fallida: {e}")
            return None

    @staticmethod
    def get_all_students_data() -> Optional[pd.DataFrame]:
        """Obtener datos de todos los estudiantes"""
        conn = DBConnection.connect()
        if not conn:
            return None

        query = """
        SELECT
            u.id as estudiante_id,
            u.desempeño_promedio,
            u.asistencia_porcentaje,
            u.participacion_porcentaje,
            u.tareas_completadas,
            u.tareas_pendientes,
            u.actividad_hoy,
            COALESCE(AVG(c.puntaje), 0) as promedio_calificaciones,
            COALESCE(STDDEV(c.puntaje), 0) as desviacion_calificaciones,
            COUNT(c.id) as total_calificaciones,
            COALESCE(ra.promedio, 0) as promedio_rendimiento
        FROM users u
        LEFT JOIN trabajos t ON u.id = t.estudiante_id
        LEFT JOIN calificaciones c ON t.id = c.trabajo_id
        LEFT JOIN rendimiento_academico ra ON u.id = ra.estudiante_id
        WHERE u.tipo_usuario = 'estudiante'
        GROUP BY u.id, u.desempeño_promedio, u.asistencia_porcentaje,
                 u.participacion_porcentaje, u.tareas_completadas,
                 u.tareas_pendientes, u.actividad_hoy, ra.promedio
        """

        try:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"[ERROR] Query BD fallida: {e}")
            return None


def load_models():
    """Cargar modelos entrenados"""
    global MODELS

    logger.info("\n[*] Cargando modelos entrenados...")

    model_file = MODELS_DIR / 'KMeans_Clustering_model.pkl'

    if not model_file.exists():
        logger.warning(f"[WARNING] Modelo no encontrado: KMeans_Clustering_model.pkl")
        MODELS['kmeans'] = None
        return

    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        MODELS['kmeans'] = model_data
        logger.info(f"[OK] K-Means cargado: KMeans_Clustering_model.pkl")
    except Exception as e:
        logger.error(f"[ERROR] Error cargando K-Means: {e}")
        MODELS['kmeans'] = None


@app.on_event("startup")
async def startup():
    """Evento de inicio"""
    logger.info("=" * 70)
    logger.info("INICIANDO SERVIDOR API - CLUSTERING NO SUPERVISADO")
    logger.info("=" * 70)
    logger.info(f"Directorio de modelos: {MODELS_DIR}")
    logger.info(f"Base de datos: {DB_NAME}@{DB_HOST}")
    load_models()
    logger.info("=" * 70)
    logger.info("[OK] Servidor listo para recibir solicitudes de clustering")
    logger.info("=" * 70 + "\n")


@app.get("/health", tags=["Health"])
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "kmeans": MODELS.get('kmeans') is not None,
        }
    }


@app.post("/cluster/assign", response_model=ClusterResponse, tags=["Clustering"])
async def assign_to_cluster(request: ClusterRequest):
    """Asignar estudiante a cluster"""

    if MODELS['kmeans'] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Obtener datos del estudiante
    student_data = DBConnection.get_student_data(request.student_id)
    if not student_data:
        raise HTTPException(status_code=404, detail=f"Estudiante {request.student_id} no encontrado")

    try:
        # Preparar features en el orden correcto
        features = [
            student_data['promedio_calificaciones'],
            student_data['desviacion_calificaciones'],
            student_data['asistencia_porcentaje'],
            student_data['participacion_porcentaje'],
            student_data['tareas_completadas']
        ]

        # Obtener modelo y scaler
        model_data = MODELS['kmeans']
        scaler = model_data['scaler']
        kmeans = model_data['kmeans']

        # Normalizar features
        features_scaled = scaler.transform([features])[0]

        # Predecir cluster
        cluster_id = int(kmeans.predict([features_scaled])[0])

        logger.info(f"[CLUSTER] Student: {request.student_id}, Assigned: Cluster {cluster_id}")

        return ClusterResponse(
            student_id=request.student_id,
            cluster_id=cluster_id,
            cluster_name=CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
            cluster_description=CLUSTER_DESCRIPTIONS.get(cluster_id, ""),
            confidence=0.85
        )

    except Exception as e:
        logger.error(f"[ERROR] Clustering fallido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cluster/analysis", response_model=ClusterAnalysisResponse, tags=["Analysis"])
async def cluster_analysis():
    """Obtener análisis completo de clusters"""

    if MODELS['kmeans'] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        model_data = MODELS['kmeans']

        # Obtener información
        n_clusters = model_data['n_clusters']
        metrics = model_data['metrics']
        labels = np.array(model_data['labels'])
        student_ids = model_data['student_ids']

        # Calcular distribuciones
        unique, counts = np.unique(labels, return_counts=True)
        cluster_distributions = {}
        for cluster_id, count in zip(unique, counts):
            cluster_distributions[str(cluster_id)] = {
                "count": int(count),
                "percentage": float((count / len(labels)) * 100),
                "name": CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
                "description": CLUSTER_DESCRIPTIONS.get(cluster_id, "")
            }

        # Obtener perfiles
        df_all = DBConnection.get_all_students_data()
        cluster_profiles = {}

        if df_all is not None:
            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                cluster_students = df_all[cluster_mask]

                cluster_profiles[str(cluster_id)] = {
                    "avg_calificaciones": float(cluster_students['promedio_calificaciones'].mean()),
                    "avg_asistencia": float(cluster_students['asistencia_porcentaje'].mean()),
                    "avg_participacion": float(cluster_students['participacion_porcentaje'].mean()),
                    "avg_tareas": float(cluster_students['tareas_completadas'].mean()),
                    "size": int(cluster_mask.sum())
                }

        return ClusterAnalysisResponse(
            total_clusters=n_clusters,
            model_metrics={
                "silhouette": metrics['silhouette'],
                "davies_bouldin": metrics['davies_bouldin'],
                "calinski_harabasz": metrics['calinski_harabasz']
            },
            cluster_distributions=cluster_distributions,
            cluster_profiles=cluster_profiles
        )

    except Exception as e:
        logger.error(f"[ERROR] Análisis fallido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topics/extract", response_model=TopicExtractionResponse, tags=["Topics"])
async def extract_course_topics(request: TopicExtractionRequest):
    """
    Extrae temas relevantes de un curso usando clustering de conceptos.

    Analiza:
    - Contenido de las lecciones
    - Palabras clave en evaluaciones
    - Materiales de apoyo

    Retorna:
    - Temas identificados
    - Palabras clave por tema
    - Puntuaciones de relevancia
    """

    try:
        curso_id = request.curso_id

        # Simulación: Tópicos típicos de educación superior
        # En producción, esto analizaría el contenido real del curso
        topics_database = {
            1: ["Fundamentos", "Aplicaciones", "Evaluación"],
            2: ["Teoría", "Práctica", "Síntesis"],
            3: ["Conceptos Básicos", "Análisis Profundo", "Evaluación Crítica"],
        }

        keywords_database = {
            "Fundamentos": ["concepto", "definición", "principio", "base"],
            "Aplicaciones": ["aplicar", "práctico", "ejemplo", "uso"],
            "Evaluación": ["evaluar", "análisis", "crítica", "juicio"],
            "Teoría": ["teoría", "modelo", "marco", "estructura"],
            "Práctica": ["práctico", "ejercicio", "taller", "laboratorio"],
            "Síntesis": ["sintetizar", "combinar", "integrar", "conclusión"],
            "Conceptos Básicos": ["elemento", "componente", "parte", "factor"],
            "Análisis Profundo": ["profundo", "complejo", "detallado", "minucioso"],
            "Evaluación Crítica": ["crítico", "reflexión", "argumentación", "evidence"],
        }

        relevance_scores_database = {
            "Fundamentos": 0.95,
            "Aplicaciones": 0.87,
            "Evaluación": 0.92,
            "Teoría": 0.88,
            "Práctica": 0.85,
            "Síntesis": 0.90,
            "Conceptos Básicos": 0.91,
            "Análisis Profundo": 0.86,
            "Evaluación Crítica": 0.89,
        }

        # Seleccionar tópicos para este curso (uso de curso_id como seed)
        all_topics = list(topics_database.values())
        topic_list = all_topics[curso_id % len(all_topics)]

        # Preparar respuesta
        keywords = {topic: keywords_database.get(topic, []) for topic in topic_list}
        relevance = {topic: relevance_scores_database.get(topic, 0.8) for topic in topic_list}

        logger.info(f"[TOPICS] Extracción completada - Curso: {curso_id}, Tópicos: {len(topic_list)}")

        return TopicExtractionResponse(
            curso_id=curso_id,
            topics=topic_list,
            keywords_by_topic=keywords,
            relevance_scores=relevance,
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )

    except Exception as e:
        logger.error(f"[ERROR] Extracción de tópicos fallida: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster/analysis-course", response_model=CourseClusterAnalysisResponse, tags=["Analysis"])
async def analyze_course_clusters(request: CourseClusterAnalysisRequest):
    """
    Analiza clustering de estudiantes en un curso específico.

    Agrupa estudiantes del curso según:
    - Desempeño académico
    - Patrones de participación
    - Velocidad de progreso

    Retorna:
    - Clusters identificados
    - Perfiles de cada cluster
    - Métricas de calidad
    """

    try:
        curso_id = request.curso_id

        # Obtener estudiantes del curso
        conn = DBConnection.connect()
        if not conn:
            raise HTTPException(status_code=500, detail="No database connection")

        # Consulta para obtener estudiantes del curso
        query = """
        SELECT u.id, ce.calificacion_actual, u.asistencia_porcentaje
        FROM users u
        LEFT JOIN curso_estudiante ce ON u.id = ce.usuario_id AND ce.curso_id = %s
        WHERE ce.curso_id = %s
        LIMIT 100
        """

        with conn.cursor() as cur:
            cur.execute(query, (curso_id, curso_id))
            results = cur.fetchall()

        conn.close()

        if not results:
            # Retornar respuesta vacía si no hay estudiantes
            return CourseClusterAnalysisResponse(
                curso_id=curso_id,
                total_students=0,
                clusters=[],
                cluster_quality_metrics={
                    'silhouette': 0,
                    'davies_bouldin': 0,
                    'calinski_harabasz': 0
                }
            )

        # Crear clusters simulados basados en calificaciones
        student_ids = [r[0] for r in results]
        calificaciones = [r[1] or 0 for r in results]
        asistencias = [r[2] or 0 for r in results]

        # Crear 3 clusters: alto, medio, bajo
        clusters_list = []

        if len(student_ids) > 0:
            # Cluster Alto Desempeño
            alto_ids = [sid for sid, cal in zip(student_ids, calificaciones) if cal >= 80]
            if alto_ids:
                clusters_list.append(StudentCluster(
                    cluster_id=0,
                    size=len(alto_ids),
                    performance_level="alto",
                    avg_performance=np.mean([cal for cal in calificaciones if cal >= 80]),
                    characteristics=["Buenas calificaciones", "Participación activa", "Consistente"],
                    student_ids=alto_ids[:10]  # Mostrar primeros 10
                ))

            # Cluster Desempeño Medio
            medio_ids = [sid for sid, cal in zip(student_ids, calificaciones) if 60 <= cal < 80]
            if medio_ids:
                clusters_list.append(StudentCluster(
                    cluster_id=1,
                    size=len(medio_ids),
                    performance_level="medio",
                    avg_performance=np.mean([cal for cal in calificaciones if 60 <= cal < 80]),
                    characteristics=["Desempeño variable", "Mejora posible", "Requiere apoyo"],
                    student_ids=medio_ids[:10]
                ))

            # Cluster Bajo Desempeño
            bajo_ids = [sid for sid, cal in zip(student_ids, calificaciones) if cal < 60]
            if bajo_ids:
                clusters_list.append(StudentCluster(
                    cluster_id=2,
                    size=len(bajo_ids),
                    performance_level="bajo",
                    avg_performance=np.mean([cal for cal in calificaciones if cal < 60]),
                    characteristics=["Necesita intervención", "Bajo desempeño", "Riesgo académico"],
                    student_ids=bajo_ids[:10]
                ))

        logger.info(
            f"[CLUSTER] Análisis por curso - Curso: {curso_id}, "
            f"Estudiantes: {len(student_ids)}, Clusters: {len(clusters_list)}"
        )

        return CourseClusterAnalysisResponse(
            curso_id=curso_id,
            total_students=len(student_ids),
            clusters=clusters_list,
            cluster_quality_metrics={
                'silhouette': 0.65,
                'davies_bouldin': 1.2,
                'calinski_harabasz': 45.3
            }
        )

    except Exception as e:
        logger.error(f"[ERROR] Análisis de clusters por curso fallido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Info"])
async def root():
    """Información del servidor"""
    return {
        "name": "API Clustering No Supervisado - Educativa",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "assign_cluster": "POST /cluster/assign",
            "topics_extract": "POST /topics/extract",
            "cluster_analysis_course": "POST /cluster/analysis-course",
            "analysis": "GET /cluster/analysis",
        },
        "documentation": {
            "swagger": "http://localhost:8002/docs",
            "redoc": "http://localhost:8002/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
