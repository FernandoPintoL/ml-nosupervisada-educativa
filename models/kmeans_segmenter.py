"""
K-Means Clustering Segmenter
Plataforma Educativa ML

Segmenta estudiantes en grupos homogéneos basado en características académicas.
Útil para:
- Identificar perfiles de estudiantes
- Targeting de intervenciones educativas
- Análisis de patrones de desempeño

Uso:
    from no_supervisado.models.kmeans_segmenter import KMeansSegmenter

    segmenter = KMeansSegmenter(n_clusters=3)
    segmenter.train(X)
    labels = segmenter.predict(X)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .base_unsupervised_model import UnsupervisedBaseModel

logger = logging.getLogger(__name__)


class KMeansSegmenter(UnsupervisedBaseModel):
    """
    Segmentador de estudiantes usando K-Means Clustering.

    Características:
    - Segmenta estudiantes en 2-5 clusters
    - Calcula métricas de calidad (silhouette, Davies-Bouldin)
    - Soporta elbow method para encontrar k óptimo
    - Genera perfiles de clusters
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Inicializar K-Means Segmenter.

        Args:
            n_clusters (int): Número de clusters (default 3)
            random_state (int): Para reproducibilidad
        """
        super().__init__(
            name="kmeans_segmenter",
            model_type="no_supervisado_clustering"
        )

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.inertia_history = []
        self.silhouette_history = []

        # Crear modelo
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            verbose=0
        )

        self.metadata['n_clusters'] = n_clusters

    def train(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Entrenar K-Means Clustering.

        Args:
            X (np.ndarray): Features de entrenamiento
            **kwargs: Argumentos adicionales

        Retorna:
            Dict: Métricas de entrenamiento
        """
        try:
            logger.info(f"Entrenando {self.name} con {self.n_clusters} clusters...")

            # Normalizar datos
            X_scaled = self.scaler.fit_transform(X)

            # Entrenar modelo
            self.model.fit(X_scaled)

            # Calcular métricas
            labels = self.model.labels_
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

            metrics = {
                'inertia': float(self.model.inertia_),
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski_harabasz),
                'n_clusters': self.n_clusters,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }

            logger.info(f"✓ Entrenamiento completado")
            logger.info(f"  Inertia: {metrics['inertia']:.4f}")
            logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            logger.info(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")

            # Actualizar metadata
            self.is_trained = True
            self.metadata['trained'] = True
            self.metadata['trained_at'] = np.datetime64('now').astype('datetime64[s]').astype(str)
            self.metadata['n_samples'] = len(X)
            self.metadata['metrics'] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error entrenando K-Means: {str(e)}")
            return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predecir cluster para muestras.

        Args:
            X (np.ndarray): Features para predicción

        Retorna:
            np.ndarray: Etiquetas de cluster (0, 1, 2, ...)
        """
        if not self.is_trained:
            logger.error("Modelo no entrenado")
            return np.array([])

        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"Error prediciendo: {str(e)}")
            return np.array([])

    def get_distances_to_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener distancias de cada muestra a todos los centros.

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Matriz de distancias (n_samples, n_clusters)
        """
        if not self.is_trained:
            logger.error("Modelo no entrenado")
            return np.array([])

        try:
            X_scaled = self.scaler.transform(X)
            return self.model.transform(X_scaled)
        except Exception as e:
            logger.error(f"Error calculando distancias: {str(e)}")
            return np.array([])

    def get_membership_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Obtener probabilidad de pertenencia a cada cluster (basada en inverso de distancia).

        Args:
            X (np.ndarray): Features

        Retorna:
            np.ndarray: Matriz de probabilidades (n_samples, n_clusters)
        """
        distances = self.get_distances_to_centers(X)
        if distances.size == 0:
            return np.array([])

        # Invertir distancias (menor distancia = mayor probabilidad)
        # Agregar pequeño epsilon para evitar división por cero
        eps = 1e-10
        inv_distances = 1.0 / (distances + eps)

        # Normalizar a probabilidades
        probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)
        return probabilities

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 6)) -> Dict[int, float]:
        """
        Encontrar número óptimo de clusters usando elbow method.

        Args:
            X (np.ndarray): Features de entrenamiento
            k_range (range): Rango de k a probar (default 2-6)

        Retorna:
            Dict: {k: silhouette_score}
        """
        logger.info(f"Buscando k óptimo en rango {k_range.start}-{k_range.stop}...")

        X_scaled = self.scaler.fit_transform(X)
        scores = {}

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                scores[k] = score

                logger.info(f"  k={k}: silhouette={score:.4f}")
            except Exception as e:
                logger.error(f"Error con k={k}: {str(e)}")

        return scores

    def get_cluster_profiles(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[int, Dict]:
        """
        Obtener perfil de cada cluster (media de features).

        Args:
            X (np.ndarray): Features
            feature_names (List[str]): Nombres de features

        Retorna:
            Dict: {cluster_id: {feature: mean_value}}
        """
        if not self.is_trained:
            logger.error("Modelo no entrenado")
            return {}

        try:
            labels = self.predict(X)

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            profiles = {}
            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                if mask.sum() > 0:
                    cluster_data = X[mask]
                    profiles[cluster_id] = {
                        'size': int(mask.sum()),
                        'percentage': float((mask.sum() / len(X)) * 100),
                        'features': {
                            feature_names[i]: float(cluster_data[:, i].mean())
                            for i in range(len(feature_names))
                        }
                    }

            return profiles

        except Exception as e:
            logger.error(f"Error calculando perfiles: {str(e)}")
            return {}

    def get_cluster_descriptions(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[int, str]:
        """
        Obtener descripción textual de cada cluster.

        Args:
            X (np.ndarray): Features
            feature_names (List[str]): Nombres de features

        Retorna:
            Dict: {cluster_id: descripción_texto}
        """
        profiles = self.get_cluster_profiles(X, feature_names)
        descriptions = {}

        for cluster_id, profile in profiles.items():
            size = profile['size']
            percentage = profile['percentage']
            descriptions[cluster_id] = f"Cluster {cluster_id}: {size} estudiantes ({percentage:.1f}%)"

        return descriptions

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de entrenamiento almacenadas.

        Retorna:
            Dict: Métricas del modelo
        """
        return self.metadata.get('metrics', {})

    def __repr__(self) -> str:
        status = "✓ Entrenado" if self.is_trained else "✗ Sin entrenar"
        return f"{self.__class__.__name__}(clusters={self.n_clusters}, {status})"
