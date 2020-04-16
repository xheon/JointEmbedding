from typing import Tuple

import scipy.spatial
import numpy as np


def compute_distance_matrix(embeddings: np.array) -> np.array:
    condensed = scipy.spatial.distance.pdist(embeddings, "euclidean")
    matrix = scipy.spatial.distance.squareform(condensed)

    return matrix


def compute_knn_confusions(distance_matrix: np.array, domains: np.array, num_neighbors: int) -> Tuple[np.array, Tuple[np.array, np.array]]:
    knn_distances, knn_indices = compute_knn(distance_matrix, num_neighbors)
    knn_domains = np.take(domains, knn_indices)
    confusions, confusions_if_scan, confusions_if_cad = compute_domain_confusion(knn_domains, domains)

    return confusions, (confusions_if_scan, confusions_if_cad)


def compute_knn(distance_matrix: np.array, k: int = 100) -> Tuple[np.array, np.array]:
    k += 1  # k nearest neighbors + the element itself
    k_i = distance_matrix.argpartition(k, axis=0)
    k_d = np.take_along_axis(distance_matrix, k_i, axis=0)
    sorted_indices = k_d.argsort(axis=0)
    k_i_sorted = np.take_along_axis(k_i, sorted_indices, axis=0)[1:k]
    k_d_sorted = np.take_along_axis(distance_matrix, k_i_sorted, axis=0)

    knn_indices = np.transpose(k_i_sorted)
    knn_distances = np.transpose(k_d_sorted)

    return knn_distances, knn_indices


def compute_domain_confusion(knn_domains: np.array, domains: np.array) -> Tuple[np.array, np.array, np.array]:
    num_scans = compute_number_per_domain(knn_domains, 0)
    num_cads = compute_number_per_domain(knn_domains, 1)

    k = knn_domains.shape[1]
    confusions = np.where(domains == 0, num_cads, num_scans) / k

    confusions_with_scan = num_cads[domains == 0] / k
    confusions_with_cad = num_scans[domains == 1] / k

    return confusions, confusions_with_scan, confusions_with_cad


def compute_number_per_domain(knn_domains: np.array, domain_label: int) -> np.array:
    domains_filtered = knn_domains == domain_label
    num_domain = np.sum(domains_filtered, axis=1)

    return num_domain
