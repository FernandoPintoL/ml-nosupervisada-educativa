"""
Data Loader para Clustering
Plataforma Educativa ML

Carga datos de estudiantes desde BD para análisis de clustering.
Selecciona features relevantes para segmentación.

Uso:
    from no_supervisado.data.cluster_loader import ClusterDataLoader

    loader = ClusterDataLoader()
    data, features = loader.load_data(limit=50)
"""

import logging
import sys
import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

# Agregar supervisado al path para usar shared
supervisado_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'supervisado')
if supervisado_dir not in sys.path:
    sys.path.insert(0, supervisado_dir)

from supervisado.data.data_loader_adapted import DataLoaderAdapted

logger = logging.getLogger(__name__)


class ClusterDataLoader:
    """
    Cargador de datos para clustering de estudiantes.

    Selecciona features que representen:
    - Desempeño académico
    - Consistencia
    - Asistencia
    - Participación
    """

    # Features seleccionadas para clustering
    CLUSTERING_FEATURES = [
        'promedio_calificaciones',      # Desempeño general
        'desviacion_notas',             # Variabilidad
        'asistencia_promedio',          # Consistencia de asistencia
        'tareas_completadas_porcentaje', # Responsabilidad
        'participacion_promedio',       # Engagement
    ]

    def __init__(self):
        """Inicializar loader de clustering."""
        self.loader = DataLoaderAdapted()
        logger.info("✓ ClusterDataLoader inicializado")

    def load_data(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Cargar datos para clustering.

        Args:
            limit (int): Límite de estudiantes a cargar

        Retorna:
            Tuple[pd.DataFrame, List[str]]: (datos, nombres de features)
        """
        try:
            logger.info("Cargando datos para clustering...")

            # Cargar datos básicos
            data, available_features = self.loader.load_training_data(limit=limit)

            if data.empty:
                logger.error("No hay datos disponibles")
                return pd.DataFrame(), []

            logger.info(f"Datos cargados: {data.shape}")

            # Seleccionar solo features disponibles
            features_to_use = [f for f in self.CLUSTERING_FEATURES if f in available_features]

            if not features_to_use:
                logger.warning("No hay features de clustering disponibles")
                logger.warning(f"Features disponibles: {available_features}")
                # Usar las primeras features numéricas disponibles
                features_to_use = [f for f in available_features if f != 'estudiante_id'][:5]

            logger.info(f"Features seleccionadas: {features_to_use}")

            # Seleccionar columnas
            cluster_data = data[features_to_use].copy()

            # Llenar NaN con media
            cluster_data = cluster_data.fillna(cluster_data.mean())

            logger.info(f"Datos preparados: {cluster_data.shape[0]} muestras, {len(features_to_use)} features")

            return cluster_data, features_to_use

        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return pd.DataFrame(), []

    def load_data_with_ids(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Cargar datos incluye IDs de estudiantes.

        Args:
            limit (int): Límite de estudiantes

        Retorna:
            Tuple: (datos, ids, nombres de features)
        """
        try:
            logger.info("Cargando datos con IDs...")

            data, features = self.loader.load_training_data(limit=limit)

            if data.empty:
                return pd.DataFrame(), pd.Series(), []

            # Guardar IDs
            student_ids = data['estudiante_id'].copy()

            # Seleccionar features de clustering
            features_to_use = [f for f in self.CLUSTERING_FEATURES if f in features]
            if not features_to_use:
                features_to_use = [f for f in features if f != 'estudiante_id'][:5]

            cluster_data = data[features_to_use].copy()
            cluster_data = cluster_data.fillna(cluster_data.mean())

            logger.info(f"Datos con IDs: {len(student_ids)} estudiantes")

            return cluster_data, student_ids, features_to_use

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return pd.DataFrame(), pd.Series(), []

    def get_feature_stats(self, data: pd.DataFrame) -> dict:
        """
        Obtener estadísticas de features.

        Args:
            data (pd.DataFrame): Datos

        Retorna:
            dict: Estadísticas por feature
        """
        stats = {}

        for col in data.columns:
            stats[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median())
            }

        return stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            self.loader.close()
        except Exception as e:
            logger.error(f"Error cerrando conexión: {str(e)}")
