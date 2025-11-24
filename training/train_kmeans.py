"""
Script de Entrenamiento: K-Means Segmentador
Plataforma Educativa ML

Entrena modelo K-Means para segmentación de estudiantes.

Uso (desde ml_educativas/):
    python -m no_supervisado.training.train_kmeans
    python -m no_supervisado.training.train_kmeans --limit 100
    python -m no_supervisado.training.train_kmeans --n-clusters 4
    python -m no_supervisado.training.train_kmeans --find-optimal-k

Uso (desde cualquier lado):
    python ml_educativas/no_supervisado/training/train_kmeans.py --limit 50
"""

import sys
import os
import logging
import argparse
from typing import Optional

import numpy as np
import pandas as pd

# Agregar ml_educativas al path
current_file = os.path.abspath(__file__)
no_supervisado_dir = os.path.dirname(os.path.dirname(current_file))
ml_educativas_dir = os.path.dirname(no_supervisado_dir)

if ml_educativas_dir not in sys.path:
    sys.path.insert(0, ml_educativas_dir)

from shared.database.connection import test_connection
from shared.config import DEBUG, LOG_LEVEL, MODELS_DIR
from no_supervisado.data.cluster_loader import ClusterDataLoader
from no_supervisado.models.kmeans_segmenter import KMeansSegmenter

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_kmeans_model(n_clusters: int = 3,
                       limit: Optional[int] = None,
                       save_model: bool = True,
                       find_optimal_k: bool = False) -> Optional[KMeansSegmenter]:
    """
    Entrenar modelo K-Means Segmenter.

    Args:
        n_clusters (int): Número de clusters (default 3)
        limit (int): Límite de estudiantes a cargar
        save_model (bool): Si True, guardar modelo después de entrenar
        find_optimal_k (bool): Si True, encontrar k óptimo primero

    Retorna:
        KMeansSegmenter: Modelo entrenado
    """
    try:
        logger.info("="*70)
        logger.info("ENTRENAMIENTO: K-MEANS SEGMENTER")
        logger.info("="*70)

        # 1. VERIFICAR CONEXIÓN
        logger.info("\n[1/5] Verificando conexión a base de datos...")
        if not test_connection():
            logger.error("✗ No se pudo conectar a la base de datos")
            return None

        # 2. CARGAR DATOS
        logger.info("\n[2/5] Cargando datos...")
        with ClusterDataLoader() as loader:
            data, features = loader.load_data(limit=limit)

        if data.empty:
            logger.error("✗ No hay datos disponibles")
            return None

        logger.info(f"Datos cargados: {data.shape}")
        logger.info(f"Features: {features}")

        # Convertir a numpy array
        X = data.values if hasattr(data, 'values') else data

        # 3. ENCONTRAR K ÓPTIMO (OPCIONAL)
        if find_optimal_k:
            logger.info("\n[3/5] Buscando k óptimo...")

            segmenter_temp = KMeansSegmenter(n_clusters=2)
            scores = segmenter_temp.find_optimal_k(X, k_range=range(2, 7))

            logger.info("\nResultados de búsqueda de k óptimo:")
            best_k = max(scores, key=scores.get)
            for k, score in sorted(scores.items()):
                marker = " ← ÓPTIMO" if k == best_k else ""
                logger.info(f"  k={k}: silhouette={score:.4f}{marker}")

            n_clusters = best_k
            logger.info(f"\nUsando k óptimo: {n_clusters}")
        else:
            logger.info(f"\n[3/5] Usando n_clusters={n_clusters}")

        # 4. ENTRENAR MODELO
        logger.info(f"\n[4/5] Entrenando K-Means con {n_clusters} clusters...")

        segmenter = KMeansSegmenter(n_clusters=n_clusters)
        segmenter.set_features(features)

        metrics = segmenter.train(X)

        if not segmenter.is_trained:
            logger.error("✗ Error durante entrenamiento")
            return None

        logger.info(f"\nMétricas de entrenamiento:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        # 5. ANALIZAR CLUSTERS
        logger.info(f"\n[5/5] Analizando clusters...")

        labels = segmenter.predict(X)
        cluster_sizes = segmenter.get_cluster_sizes(labels)
        cluster_dist = segmenter.get_cluster_distribution(labels)

        logger.info(f"\nDistribución de clusters:")
        for cluster_id in range(n_clusters):
            size = cluster_sizes.get(cluster_id, 0)
            percentage = cluster_dist.get(cluster_id, 0)
            logger.info(f"  Cluster {cluster_id}: {size} estudiantes ({percentage:.1f}%)")

        # Perfiles de clusters
        logger.info(f"\nPerfiles de clusters:")
        profiles = segmenter.get_cluster_profiles(X, features)
        for cluster_id, profile in profiles.items():
            logger.info(f"\n  Cluster {cluster_id}:")
            logger.info(f"    Tamaño: {profile['size']} ({profile['percentage']:.1f}%)")
            logger.info(f"    Features promedio:")
            for feature, value in profile['features'].items():
                logger.info(f"      {feature}: {value:.2f}")

        # 6. GUARDAR MODELO
        logger.info("\n[6/6] Guardando modelo...")
        if save_model:
            filepath = segmenter.save(directory=MODELS_DIR)
            logger.info(f"✓ Modelo guardado en: {filepath}")
        else:
            logger.info("Modelo no guardado (--save-model no activado)")

        logger.info("\n" + "="*70)
        logger.info("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*70)

        return segmenter

    except Exception as e:
        logger.error(f"✗ Error durante entrenamiento: {str(e)}", exc_info=True)
        return None


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Entrenamiento de K-Means Segmenter'
    )
    parser.add_argument('--n-clusters', type=int, default=3,
                       help='Número de clusters (default: 3)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Límite de estudiantes a cargar')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Guardar modelo después de entrenar')
    parser.add_argument('--find-optimal-k', action='store_true', default=False,
                       help='Encontrar k óptimo antes de entrenar')

    args = parser.parse_args()

    # Entrenar modelo
    segmenter = train_kmeans_model(
        n_clusters=args.n_clusters,
        limit=args.limit,
        save_model=args.save_model,
        find_optimal_k=args.find_optimal_k
    )

    if segmenter is None:
        exit(1)

    exit(0)


if __name__ == '__main__':
    main()
