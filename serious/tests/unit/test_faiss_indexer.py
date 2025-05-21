# tests/unit/test_faiss_indexer.py

import pytest
import numpy as np
import faiss # FAISS must be installed

# Add src to Python path
import sys
import os
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.core.faiss_m1 import FAISSIndexer

DIM = 32 # Using a smaller dimension for tests
M_PQ = 4 # Number of subquantizers for PQ
NBITS_PQ = 8 # Bits per subquantizer
K_CENTROIDS_PER_SUBQUANTIZER = 2**NBITS_PQ # This is 256

@pytest.fixture
def dummy_embeddings():
    # Generate enough vectors for "good" PQ training
    # Heuristic: M * 256 or more. Let's use slightly more than K_CENTROIDS_PER_SUBQUANTIZER * M_PQ
    # Or simply a fixed larger number that's definitely sufficient for full PQ training.
    # num_vectors = K_CENTROIDS_PER_SUBQUANTIZER * M_PQ + 100 # e.g., 256*4 + 100 = 1124
    # A common rule of thumb for training IndexPQ is to have at least k_centroids * 39 training vectors for good results.
    # For IndexPQ, training points `N` should be `N > k` (where `k` is `dsub_Ncentroids = 1 << (M*nbits_per_code)` for IVFPQ)
    # For PQ standalone, it's about training each sub-quantizer. Each has K_CENTROIDS_PER_SUBQUANTIZER.
    # So, we need at least K_CENTROIDS_PER_SUBQUANTIZER training points.
    # Let's make dummy_embeddings definitely sufficient
    num_vectors = K_CENTROIDS_PER_SUBQUANTIZER * 2 # e.g., 512, enough to train one subquantizer well
    if num_vectors < 1000: # Ensure it's also a decent size overall for better PQ
        num_vectors = 1000
    return np.random.rand(num_vectors, DIM).astype('float32')

@pytest.fixture
def small_dummy_embeddings_for_warning():
    # Enough to pass FAISS's hard requirement (>= K_CENTROIDS_PER_SUBQUANTIZER)
    # but less than the "good quality" heuristic (e.g., M_PQ * K_CENTROIDS_PER_SUBQUANTIZER)
    # Your heuristic is: data.shape[0] < self.index.pq.M * 256
    # So we need: K_CENTROIDS_PER_SUBQUANTIZER <= num_vectors < M_PQ * K_CENTROIDS_PER_SUBQUANTIZER
    # Example: 256 <= num_vectors < 4 * 256 (1024)
    # Let's pick a value like K_CENTROIDS_PER_SUBQUANTIZER + 10, e.g. 266
    num_vectors = K_CENTROIDS_PER_SUBQUANTIZER + 10 # e.g., 256 + 10 = 266
    return np.random.rand(num_vectors, DIM).astype('float32')

@pytest.fixture
def very_small_dummy_embeddings_for_error(): # Not enough for FAISS's hard requirement
    # This should cause the RuntimeError from FAISS.
    # num_vectors < K_CENTROIDS_PER_SUBQUANTIZER
    num_vectors = K_CENTROIDS_PER_SUBQUANTIZER - 1 # e.g., 255
    if num_vectors <=0: num_vectors = 1 # handle case where K_CENTROIDS_PER_SUBQUANTIZER might be 1
    return np.random.rand(num_vectors, DIM).astype('float32')


@pytest.fixture
def query_embeddings():
    return np.random.rand(5, DIM).astype('float32')

def test_faiss_indexer_initialization():
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    assert indexer.dim == DIM
    assert isinstance(indexer.index, faiss.IndexPQ)
    assert not indexer.is_trained

def test_faiss_indexer_build_and_train(dummy_embeddings):
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    indexer.build(dummy_embeddings)
    assert indexer.is_trained
    assert indexer.index.ntotal == dummy_embeddings.shape[0]

# This test is now for the warning, FAISS should still train.
def test_faiss_indexer_build_triggers_low_training_data_warning(small_dummy_embeddings_for_warning, capsys):
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    # The warning condition is: data.shape[0] < self.index.pq.M * 256
    # small_dummy_embeddings_for_warning (266) < M_PQ (4) * 256 (1024) -> True, so warning should appear.
    # FAISS's hard requirement: data.shape[0] (266) >= K_CENTROIDS_PER_SUBQUANTIZER (256) -> True, so FAISS trains.
    indexer.build(small_dummy_embeddings_for_warning)
    captured = capsys.readouterr()
    assert "Warning: Low number of training vectors" in captured.out
    assert f"({small_dummy_embeddings_for_warning.shape[0]})" in captured.out # Check the number
    assert indexer.is_trained # It still trains
    assert indexer.index.ntotal == small_dummy_embeddings_for_warning.shape[0]

# New test for the FAISS RuntimeError
def test_faiss_indexer_build_raises_runtime_error_for_critically_insufficient_data(very_small_dummy_embeddings_for_error):
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    # very_small_dummy_embeddings_for_error (255) < K_CENTROIDS_PER_SUBQUANTIZER (256) -> FAISS RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        indexer.build(very_small_dummy_embeddings_for_error)
    assert "Number of training points" in str(excinfo.value)
    assert "should be at least as large as number of clusters" in str(excinfo.value)
    # The warning from our code might or might not print before FAISS crashes,
    # depending on where FAISS checks. Let's assume our warning will print.
    # To test this reliably, you might need to check capsys *before* the assertRaises context.
    # However, the primary goal here is to ensure FAISS's error is caught.

def test_faiss_indexer_build_empty_embeddings(capsys):
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    empty_embeds = np.array([], dtype='float32').reshape(0, DIM)
    indexer.build(empty_embeds)
    captured = capsys.readouterr()
    assert "No embeddings provided to build the index." in captured.out
    assert not indexer.is_trained # Should not train if no data
    assert indexer.index.ntotal == 0

def test_faiss_indexer_search_after_build(dummy_embeddings, query_embeddings):
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    indexer.build(dummy_embeddings)
    
    k = 5
    distances, indices = indexer.search(query_embeddings, k=k)
    assert distances.shape == (query_embeddings.shape[0], k)
    assert indices.shape == (query_embeddings.shape[0], k)

def test_faiss_indexer_search_single_query_vector(dummy_embeddings):
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    indexer.build(dummy_embeddings)
    
    single_query = np.random.rand(DIM).astype('float32')
    k = 3
    distances, indices = indexer.search(single_query, k=k)
    assert distances.shape == (1, k) # Should reshape internally
    assert indices.shape == (1, k)

def test_faiss_indexer_search_on_empty_index(query_embeddings, capsys):
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    # Index is not built, or built with no data
    distances, indices = indexer.search(query_embeddings, k=5)
    captured = capsys.readouterr()
    assert "Index is empty. Cannot perform search." in captured.out
    assert distances.size == 0 # Check for empty array
    assert indices.size == 0

def test_faiss_indexer_index_not_trained_but_add_called_on_empty(dummy_embeddings, query_embeddings, capsys):
    # This tests the search path where is_trained might be true (e.g. from a previous build),
    # but ntotal is 0 (e.g. index was reset or add was never called with data).
    indexer = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    
    # Simulate index being trained (e.g., on a small subset that doesn't add to ntotal directly for this test)
    # Or, more simply, just set the flag and ensure ntotal is 0.
    # indexer.index.train(dummy_embeddings[:K_CENTROIDS_PER_SUBQUANTIZER + 1]) # Train on a minimal valid set
    # indexer.is_trained = True
    # At this point, indexer.index.ntotal is still 0 because add() hasn't been called yet in this instance.

    # A simpler way to test the "Index is empty" search condition:
    # Build with actual data, then reset the index, then try to search.
    # However, our FAISSIndexer doesn't have a reset method.
    # So, the current `test_faiss_indexer_search_on_empty_index` covers the scenario
    # where `build` was called with no data, or `build` was never called.

    # Let's test the case where build() *was* called, trained the index, but added no vectors.
    # This would happen if build was called with empty embeddings *after* it had already been trained once.
    # The `FAISSIndexer.build` method, if called with empty data, prints "No embeddings..." and returns.
    # So self.index.ntotal would remain from a previous call, or be 0 if first call.
    # The important check is `if not self.index.ntotal > 0:` in search.

    # Re-testing the `search_on_empty_index` is essentially what this would cover.
    # The `test_faiss_indexer_search_on_empty_index` already tests when `ntotal` is 0.
    # If we want to ensure the `is_trained` flag doesn't bypass this:
    indexer_trained_no_data = FAISSIndexer(dim=DIM, m=M_PQ, nbits=NBITS_PQ)
    # Manually train it on some data so is_trained becomes True
    minimal_train_data = np.random.rand(K_CENTROIDS_PER_SUBQUANTIZER + 5, DIM).astype('float32')
    indexer_trained_no_data.index.train(minimal_train_data)
    indexer_trained_no_data.is_trained = True
    # Now, index is trained, but ntotal is 0 because we haven't called index.add() or our wrapper's build() with data to add.
    
    distances, indices = indexer_trained_no_data.search(query_embeddings, k=5)
    captured = capsys.readouterr()
    assert "Index is empty. Cannot perform search." in captured.out
    assert distances.size == 0
    assert indices.size == 0
