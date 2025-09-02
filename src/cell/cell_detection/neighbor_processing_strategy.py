from typing import List, Tuple
from abc import ABC, abstractmethod
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import NearestNeighbors

from src.cell.cell_detection.cone import Cone

class NeighborProcessingStrategy(ABC):
    """
    Abstract base class for neighbor processing strategies.

    This class defines the interface for processing neighbors.
    """
    @abstractmethod
    def process_neighbors(self, neighbors: List[List[int]]) -> List[int]:
        """
        Process neighbors to produce a list of results.

        :param neighbors: A list of neighbor coordinates.
        :type neighbors: List[List[int]]
        :return: A list of processed neighbor results.
        :rtype: List[int]
        """
        pass

class DBSCANStrategy(NeighborProcessingStrategy):
    """
    DBSCAN strategy for processing neighbors.

    This class implements the DBSCAN clustering algorithm to process neighbors.
    """
    def process_neighbors(self, neighbors: List[List[int]], distance: float) -> List[List[int]]:
        """
        Process neighbors using DBSCAN clustering.

        This method applies the DBSCAN clustering algorithm to the given neighbors and returns the centroids of the clusters.

        :param neighbors: A list of neighbor coordinates.
        :type neighbors: List[List[int]]
        :param distance: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :type distance: float
        :return: A list of centroids of the clusters.
        :rtype: List[List[int]]
        """
        if not neighbors:
            return []
        if len(neighbors) == 1:
            return neighbors
        points = np.array(neighbors)
        clustering = DBSCAN(eps=distance, min_samples=2).fit(points)
        labels = clustering.labels_
        centroids = []
        for label in set(labels):
            if label != -1:
                cluster_points = points[labels == label]
                centroid_x = np.mean(cluster_points[:, 1])
                centroid_y = np.mean(cluster_points[:, 0])
                centroids.append([round(centroid_y), round(centroid_x)])
            else:
                noise_points = points[labels == label]
                for noise_point in noise_points:
                    centroids.append([noise_point[0], noise_point[1]])
        return centroids

class HierarchicalClusteringStrategy(NeighborProcessingStrategy):
    def process_neighbors(self, neighbors: List[List[int]], d: float) -> List[List[int]]:
        if not neighbors:
            return []
        if len(neighbors) == 1:
            return neighbors

        points = np.array(neighbors)
        linked = linkage(points, 'single')
        labels = fcluster(linked, d, criterion='distance')

        centroids = []
        for label in np.unique(labels):
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append([round(centroid[0]), round(centroid[1])])

        return centroids

# class VoronoiClusteringStrategy(NeighborProcessingStrategy):
#     def process_neighbors(self, neighbors: List[Tuple[int, int]], d: float) -> List[Tuple[int, int]]:
#         if not neighbors:
#             return []
#         if len(neighbors) == 1:
#             return neighbors

#         points = np.array(neighbors)
#         vor = Voronoi(points)

#         centroids = []
#         for region in vor.regions:
#             if not -1 in region and len(region) > 0:
#                 polygon = [vor.vertices[i] for i in region]
#                 centroid = np.mean(polygon, axis=0)
#                 centroids.append((round(centroid[0]), round(centroid[1])))
#         filtered_centroids = []
#         for centroid in centroids:
#             if any(np.linalg.norm(np.array(centroid) - point) <= d for point in points):
#                 filtered_centroids.append(centroid)

#         return filtered_centroids

class HierarchicalClusteringStrategy(NeighborProcessingStrategy):
    """
    Hierarchical clustering strategy for processing neighbors.

    This class implements the hierarchical clustering algorithm to process neighbors.
    """
    def process_neighbors(self, neighbors: List[List[int]], d: float) -> List[List[int]]:
        """
        Process neighbors using hierarchical clustering.

        This method applies the hierarchical clustering algorithm to the given neighbors and returns the centroids of the clusters.

        :param neighbors: A list of neighbor coordinates.
        :type neighbors: List[List[int]]
        :param d: The threshold to apply when forming flat clusters.
        :type d: float
        :return: A list of centroids of the clusters.
        :rtype: List[List[int]]
        """
        if not neighbors:
            return []
        if len(neighbors) == 1:
            return neighbors

        points = np.array(neighbors)
        linked = linkage(points, 'single')
        labels = fcluster(linked, d, criterion='distance')

        centroids = []
        for label in np.unique(labels):
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append([round(centroid[0]), round(centroid[1])])

        return centroids

class MathematicalDistanceBasedStrategy(NeighborProcessingStrategy):
    """
    Mathematical distance-based strategy for processing neighbors.

    This class implements a custom clustering algorithm based on pairwise distances to process neighbors.
    """
    def process_neighbors(self, neighbors: List[List[int]], d: float) -> List[List[int]]:
        """
        Process neighbors using a mathematical distance-based clustering algorithm.

        This method computes pairwise distances between neighbors, assigns cluster labels based on the distance threshold,
        and calculates the centroids of the clusters.

        :param neighbors: A list of neighbor coordinates.
        :type neighbors: List[List[int]]
        :param d: The distance threshold for clustering.
        :type d: float
        :return: A list of centroids of the clusters.
        :rtype: List[List[int]]
        """
        if not neighbors:
            return []
        points = np.array(neighbors)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Assign cluster labels
        labels = -1 * np.ones(points.shape[0])
        current_label = 0
        for i in range(points.shape[0]):
            if labels[i] == -1:
                labels[i] = current_label
                for j in range(points.shape[0]):
                    if distances[i, j] <= d:
                        labels[j] = current_label
                current_label += 1

        # Calculate centroids
        centroids = []
        for label in set(labels):
            if label != -1:
                cluster_points = points[labels == label]
                centroid_x = np.mean(cluster_points[:, 0])
                centroid_y = np.mean(cluster_points[:, 1])
                centroids.append([round(centroid_x), round(centroid_y)])
            else:
                noise_points = points[labels == label]
                for noise_point in noise_points:
                    centroids.append([noise_point[0], noise_point[1]])
        return centroids
