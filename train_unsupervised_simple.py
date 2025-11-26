#!/usr/bin/env python
"""
Script Simplificado de Entrenamiento de Modelos No Supervisados
INDEPENDIENTE - Se ejecuta completamente desde no_supervisado/
Se conecta directamente a la BD de Laravel
"""

import sys
import os
import logging
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de BD
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE', 'educativa')
DB_USER = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1234')

MODELS_DIR = Path(__file__).parent / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)


class DBConnection:
    """Conexion a la base de datos de Laravel"""

    def __init__(self):
        self.conn = None

    def connect(self) -> bool:
        """Conectar a la BD"""
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info(f"[OK] Conectado a BD: {DB_NAME}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Error conectando a BD: {e}")
            return False

    def execute_query(self, query: str) -> list:
        """Ejecutar query y retornar resultados"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"[ERROR] Error en query: {e}")
            return []

    def close(self):
        """Cerrar conexion"""
        if self.conn:
            self.conn.close()


class UnsupervisedTrainer:
    """Entrena modelos de ML no supervisados"""

    def __init__(self):
        self.db = DBConnection()
        self.models = {}
        self.scaler = StandardScaler()

    def cargar_datos(self) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
        """Cargar datos de la BD para clustering"""
        logger.info("[*] Cargando datos de la base de datos...")

        if not self.db.connect():
            return None, None

        # Query para obtener datos de estudiantes
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
        LIMIT 100
        """

        results = self.db.execute_query(query)

        if not results:
            logger.error("[ERROR] No se encontraron datos de estudiantes")
            return None, None

        df = pd.DataFrame(results)
        logger.info(f"[OK] Datos cargados: {len(df)} estudiantes")

        # Features para clustering
        features = {
            'promedio_calificaciones': 'Promedio Calificaciones',
            'desviacion_calificaciones': 'Consistencia (Desviacion)',
            'asistencia_porcentaje': 'Asistencia (%)',
            'participacion_porcentaje': 'Participacion (%)',
            'tareas_completadas': 'Tareas Completadas'
        }

        return df, features

    def entrenar_kmeans(self, n_clusters: int = 3) -> bool:
        """Entrenar modelo K-Means para clustering de estudiantes"""
        logger.info(f"\n[MODEL] Entrenando K-Means Clustering ({n_clusters} clusters)...")

        df, features_dict = self.cargar_datos()
        if df is None or df.empty:
            logger.error("[ERROR] Sin datos para entrenar")
            return False

        # Features seleccionadas
        features = list(features_dict.keys())

        # Preparar datos
        X = df[features].fillna(df[features].mean())

        if X.empty or X.shape[0] < n_clusters:
            logger.error(f"[ERROR] Datos insuficientes (necesita al menos {n_clusters} muestras)")
            return False

        logger.info(f"  Datos: {X.shape[0]} muestras, {X.shape[1]} features")

        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar K-Means
        logger.info("  Entrenando K-Means...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )

        labels = kmeans.fit_predict(X_scaled)

        # Calcular metricas
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

        logger.info("[OK] Modelo entrenado:")
        logger.info(f"  Silhouette Score: {silhouette:.4f} (mayor es mejor)")
        logger.info(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (menor es mejor)")
        logger.info(f"  Calinski-Harabasz Index: {calinski_harabasz:.4f} (mayor es mejor)")

        # Analizar distribucion de clusters
        logger.info("\n  Distribucion de clusters:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            logger.info(f"    Cluster {cluster_id}: {count} estudiantes ({percentage:.1f}%)")

        # Analizar caracteristicas de cada cluster
        logger.info("\n  Caracteristicas promedio por cluster:")
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            logger.info(f"\n    Cluster {cluster_id}:")
            for feature in features:
                mean_val = df[cluster_mask][feature].mean()
                logger.info(f"      {features_dict[feature]}: {mean_val:.2f}")

        # Guardar modelo
        model_data = {
            'kmeans': kmeans,
            'scaler': self.scaler,
            'features': features,
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'student_ids': df['estudiante_id'].tolist(),
            'metrics': {
                'silhouette': float(silhouette),
                'davies_bouldin': float(davies_bouldin),
                'calinski_harabasz': float(calinski_harabasz)
            }
        }

        model_path = MODELS_DIR / 'KMeans_Clustering_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"  Modelo guardado: {model_path}")

        self.models['kmeans'] = model_data
        return True

    def registrar_entrenamientos(self, resultados: dict) -> None:
        """Guardar registro de entrenamientos en JSON"""
        log_path = MODELS_DIR / 'training_log.json'

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'modelos': resultados,
            'directorio': str(MODELS_DIR),
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Log guardado en: {log_path}")

    def entrenar_todos(self) -> bool:
        """Entrenar todos los modelos no supervisados"""
        logger.info("=" * 70)
        logger.info("INICIANDO ENTRENAMIENTO DE MODELOS NO SUPERVISADOS")
        logger.info("=" * 70)

        resultados = {
            'K-Means Clustering': self.entrenar_kmeans(n_clusters=3),
        }

        logger.info("\n" + "=" * 70)
        logger.info("RESUMEN DE ENTRENAMIENTOS")
        logger.info("=" * 70)

        for modelo, exito in resultados.items():
            estado = "[OK]" if exito else "[ERROR]"
            logger.info(f"{estado} - {modelo}")

        logger.info("=" * 70)
        logger.info(f"Modelos guardados en: {MODELS_DIR}")

        # Registrar entrenamientos
        self.registrar_entrenamientos(resultados)

        self.db.close()

        return all(resultados.values())


def main():
    """Funcion principal"""
    try:
        trainer = UnsupervisedTrainer()
        exito = trainer.entrenar_todos()

        if exito:
            logger.info("\n[SUCCESS] Todos los modelos entrenados exitosamente!")
            logger.info(f"Modelos guardados en: {MODELS_DIR}")
            logger.info("\nArchivos generados:")
            for f in MODELS_DIR.glob("*.pkl"):
                logger.info(f"  - {f.name}")
            return 0
        else:
            logger.error("\n[FAILED] Error en el entrenamiento")
            return 1
    except Exception as e:
        logger.error(f"[ERROR] Excepcion no manejada: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
