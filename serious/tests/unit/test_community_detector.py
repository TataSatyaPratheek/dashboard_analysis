# tests/unit/test_community_detector.py

import pytest
import numpy as np
import igraph as ig # Ensure igraph is installed
from unittest.mock import MagicMock, patch, create_autospec # create_autospec is useful

# Add src to Python path
import sys
import os
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.core.gala import CommunityDetector # Renamed from gala.py in thought process
from sklearn.neighbors import NearestNeighbors # For spec in MagicMock

DIM_GALA = 16 # Small dimension for tests
N_NEIGHBORS_GALA = 3

@pytest.fixture
def sample_embeddings_gala():
    # Enough samples for n_neighbors
    # For 3 clusters
    cluster1 = np.random.rand(10, DIM_GALA) + np.array([1,1]* (DIM_GALA//2))
    cluster2 = np.random.rand(10, DIM_GALA) + np.array([-1,-1]* (DIM_GALA//2))
    cluster3 = np.random.rand(10, DIM_GALA) + np.array([1,-1]* (DIM_GALA//2))
    return np.vstack([cluster1, cluster2, cluster3]).astype('float32')

@pytest.fixture
def few_embeddings_gala(): # Less than n_neighbors
    return np.random.rand(N_NEIGHBORS_GALA - 1, DIM_GALA).astype('float32')

def test_community_detector_initialization():
    detector = CommunityDetector(n_neighbors=5, use_weights=True)
    assert detector.n_neighbors == 5
    assert detector.use_weights is True

def test_community_detector_detect_communities_no_weights(sample_embeddings_gala):
    detector = CommunityDetector(n_neighbors=N_NEIGHBORS_GALA, use_weights=False)
    communities = detector.detect(sample_embeddings_gala)
    assert isinstance(communities, ig.VertexClustering)
    assert len(communities) > 0 # Expect some communities

def test_community_detector_detect_communities_with_weights(sample_embeddings_gala):
    detector = CommunityDetector(n_neighbors=N_NEIGHBORS_GALA, use_weights=True)
    communities = detector.detect(sample_embeddings_gala)
    assert isinstance(communities, ig.VertexClustering)
    assert len(communities) > 0

def test_community_detector_insufficient_embeddings(few_embeddings_gala, capsys):
    # n_neighbors default is 15, few_embeddings_gala has N_NEIGHBORS_GALA - 1
    detector = CommunityDetector(n_neighbors=N_NEIGHBORS_GALA) 
    communities = detector.detect(few_embeddings_gala)
    assert communities is None
    captured = capsys.readouterr()
    expected_n_neighbors = N_NEIGHBORS_GALA
    assert f"Not enough embeddings ({few_embeddings_gala.shape[0]}) to build a KNN graph with {expected_n_neighbors} neighbors." in captured.out


def test_community_detector_sklearn_adj_matrix_is_matrix_type(sample_embeddings_gala, mocker):
    # This test ensures that the community detection process handles adjacency matrices
    # correctly, specifically when the adjacency matrix is of type np.matrix
    # (which might be returned by a mock or an unusual pathway).
    detector = CommunityDetector(n_neighbors=N_NEIGHBORS_GALA, use_weights=True)

    # Create a dummy adjacency matrix of type np.matrix
    # Since use_weights=True, this matrix should contain float weights.
    num_samples = sample_embeddings_gala.shape[0]
    raw_adj_data = np.random.rand(num_samples, num_samples)
    # Ensure at least one off-diagonal element is significant and will survive thresholding
    if num_samples > 1:
        raw_adj_data[0, 1] = 0.8
        # Ensure symmetry for this specific element, though full symmetrization follows
        raw_adj_data[1, 0] = 0.8
    # Symmetrize for undirected graph
    symmetrized_adj_data = (raw_adj_data + raw_adj_data.T) / 2
    # Ensure diagonal is zero if mode is 'distance' (sklearn usually ensures this)
    np.fill_diagonal(symmetrized_adj_data, 0)
    # Threshold to make it sparse-like but keep the float values for weights
    # Values <= 0.5 become 0, others (like our 0.8) remain.
    final_adj_data = np.where(symmetrized_adj_data > 0.5, symmetrized_adj_data, 0.0)
    # Ensure there are non-zero elements after all operations for the test to be meaningful
    assert np.any(final_adj_data), "Mocked adjacency data has no non-zero elements!"
    # Convert to np.matrix for the test
    np_matrix_obj = np.asmatrix(final_adj_data)
    assert np.count_nonzero(np_matrix_obj) > 0, "Mocked np.matrix has no non-zero elements!"

    # Create a mock for the NearestNeighbors *instance*
    mock_nbrs_instance = MagicMock(spec=NearestNeighbors) # Using spec is good practice
    # Configure the kneighbors_graph method of this instance
    mock_nbrs_instance.kneighbors_graph.return_value = np_matrix_obj
    # Configure the fit method to return the same instance (self)
    mock_nbrs_instance.fit.return_value = mock_nbrs_instance

    # Patch the NearestNeighbors CLASS to return your configured INSTANCE
    with patch('src.core.gala.NearestNeighbors', return_value=mock_nbrs_instance) as mock_NearestNeighbors_class:
        communities = detector.detect(sample_embeddings_gala)

    assert isinstance(communities, ig.VertexClustering)
    # Check that NearestNeighbors was instantiated as expected
    mock_NearestNeighbors_class.assert_called_once_with(
        n_neighbors=detector.n_neighbors, 
        algorithm='auto', 
        metric='euclidean', 
        n_jobs=-1
    )
    # Check that fit was called on the instance
    mock_nbrs_instance.fit.assert_called_once() 
    # Check that kneighbors_graph was called on the instance
    mock_nbrs_instance.kneighbors_graph.assert_called_once()

# Test with very few samples, e.g., 2 samples and n_neighbors=1
def test_community_detector_minimal_samples(capsys):
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
    detector = CommunityDetector(n_neighbors=1, use_weights=False)
    communities = detector.detect(embeddings)
    assert isinstance(communities, ig.VertexClustering)
    assert len(communities) <= 2 # Could be 1 or 2 communities
