"""
Clase Base Abstracta para Modelos No Supervisados
Plataforma Educativa ML

Define la interfaz común para todos los modelos no supervisados:
- K-Means Clustering
- Isolation Forest
- Hierarchical Clustering
- Collaborative Filtering
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import logging
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd

from shared.config import MODELS_DIR, DEBUG

# Configurar logger
logger = logging.getLogger(__name__)


class UnsupervisedBaseModel(ABC):
    """
    Clase base abstracta para todos los modelos no supervisados.

    Proporciona:
    - Interfaz común para train/predict/fit
    - Manejo de guardado/carga de modelos
    - Logging automático
    - Metadata de modelos
    """

    def __init__(self, name: str, model_type: str = "no_supervisado"):
        """
        Inicializar modelo no supervisado.

        Args:
            name (str): Nombre único del modelo (ej: "kmeans_segmenter")
            model_type (str): Tipo (no_supervisado, clustering, anomaly_detection)
        """
        self.name = name
        self.model_type = model_type
        self.model = None  # El modelo real (sklearn, etc)
        self.is_trained = False
        self.features = []
        self.metadata = {
            'name': name,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'trained': False,
            'trained_at': None,
            'n_clusters': None,
            'n_samples': None,
            'features_used': []
        }

        logger.info(f"✓ {self.name} inicializado")

    # ===========================================
    # MÉTODOS ABSTRACTOS
    # ===========================================

    @abstractmethod
    def train(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Entrenar el modelo no supervisado.

        Args:
            X (np.ndarray): Features de entrenamiento (n_samples, n_features)
            **kwargs: Argumentos específicos del modelo

        Retorna:
            Dict[str, Any]: Diccionario con métricas de entrenamiento
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realizar predicciones (asignación a clusters/anomalías).

        Args:
            X (np.ndarray): Features para predicción

        Retorna:
            np.ndarray: Predicciones (labels de cluster o scores)
        """
        pass

    # ===========================================
    # MÉTODOS DE GUARDADO/CARGA
    # ===========================================

    def save(self, filename: Optional[str] = None, directory: Optional[str] = None) -> str:
        """
        Guardar el modelo entrenado.

        Args:
            filename (str): Nombre del archivo (default: {name}_model.pkl)
            directory (str): Directorio donde guardar (default: MODELS_DIR)

        Retorna:
            str: Ruta completa del archivo guardado
        """
        if not self.is_trained:
            logger.warning(f"{self.name} no está entrenado, salvando de todas formas")

        # Usar defaults si no se proporciona
        if filename is None:
            filename = f"{self.name}_model.pkl"
        if directory is None:
            directory = MODELS_DIR

        filepath = f"{directory}/{filename}"

        try:
            # Actualizar metadata
            self.metadata['trained_at'] = datetime.now().isoformat()
            self.metadata['trained'] = self.is_trained
            self.metadata['file'] = filepath

            # Guardar modelo y metadata
            joblib.dump({
                'model': self.model,
                'features': self.features,
                'metadata': self.metadata
            }, filepath)

            logger.info(f"✓ {self.name} guardado en {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"✗ Error guardando {self.name}: {str(e)}")
            raise

    def load(self, filepath: str) -> bool:
        """
        Cargar un modelo entrenado.

        Args:
            filepath (str): Ruta al archivo del modelo

        Retorna:
            bool: True si se cargó exitosamente
        """
        try:
            data = joblib.load(filepath)

            self.model = data['model']
            self.features = data['features']
            self.metadata = data.get('metadata', {})
            self.is_trained = True

            logger.info(f"✓ {self.name} cargado desde {filepath}")
            return True

        except Exception as e:
            logger.error(f"✗ Error cargando {self.name}: {str(e)}")
            return False

    # ===========================================
    # MÉTODOS PARA CLUSTERING
    # ===========================================

    def get_cluster_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener etiquetas de cluster para muestras.

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Etiquetas de cluster (0, 1, 2, ...)
        """
        if not self.is_trained:
            logger.error(f"{self.name} no está entrenado")
            return np.array([])

        try:
            return self.predict(X)
        except Exception as e:
            logger.error(f"Error obteniendo etiquetas: {str(e)}")
            return np.array([])

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Obtener centros de cluster (si disponible).

        Retorna:
            np.ndarray: Centros de cluster o None
        """
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None

    def get_n_clusters(self) -> int:
        """
        Obtener número de clusters.

        Retorna:
            int: Número de clusters
        """
        if hasattr(self.model, 'n_clusters'):
            return self.model.n_clusters
        if hasattr(self.model, 'cluster_centers_'):
            return len(self.model.cluster_centers_)
        return self.metadata.get('n_clusters', 0)

    def get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """
        Obtener tamaño de cada cluster.

        Args:
            labels (np.ndarray): Etiquetas de cluster

        Retorna:
            Dict: {cluster_id: tamaño}
        """
        unique, counts = np.unique(labels, return_counts=True)
        return {int(cluster_id): int(count) for cluster_id, count in zip(unique, counts)}

    def get_cluster_distribution(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Obtener distribución porcentual de clusters.

        Args:
            labels (np.ndarray): Etiquetas de cluster

        Retorna:
            Dict: {cluster_id: porcentaje}
        """
        sizes = self.get_cluster_sizes(labels)
        total = sum(sizes.values())
        return {cluster_id: (size / total) * 100 for cluster_id, size in sizes.items()}

    # ===========================================
    # MÉTODOS PARA ANOMALÍAS
    # ===========================================

    def get_anomaly_scores(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Obtener scores de anomalía (si disponible).

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Scores de anomalía o None
        """
        if hasattr(self.model, 'score_samples'):
            try:
                return self.model.score_samples(X)
            except Exception as e:
                logger.error(f"Error obteniendo anomaly scores: {str(e)}")
                return None
        return None

    def detect_anomalies(self, X: np.ndarray, threshold: float = -0.5) -> np.ndarray:
        """
        Detectar anomalías usando threshold.

        Args:
            X (np.ndarray): Features
            threshold (float): Threshold para anomalía

        Retorna:
            np.ndarray: Array booleano de anomalías
        """
        scores = self.get_anomaly_scores(X)
        if scores is None:
            return np.array([])

        return scores < threshold

    # ===========================================
    # UTILIDADES
    # ===========================================

    def set_features(self, features: List[str]) -> None:
        """
        Establecer nombres de features.

        Args:
            features (List[str]): Lista de nombres de features
        """
        self.features = features
        self.metadata['features_used'] = features

    def get_metadata(self) -> Dict[str, Any]:
        """
        Obtener metadata del modelo.

        Retorna:
            Dict[str, Any]: Metadata del modelo
        """
        return self.metadata

    def get_training_info(self) -> Dict[str, Any]:
        """
        Obtener información de entrenamiento.

        Retorna:
            Dict: Información de entrenamiento
        """
        return {
            'name': self.name,
            'model_type': self.model_type,
            'trained': self.is_trained,
            'trained_at': self.metadata.get('trained_at'),
            'n_clusters': self.get_n_clusters(),
            'n_samples': self.metadata.get('n_samples'),
            'features': self.features
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, trained={self.is_trained}, clusters={self.get_n_clusters()})"
