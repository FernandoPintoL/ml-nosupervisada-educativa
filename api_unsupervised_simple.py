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
