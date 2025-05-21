# tests/unit/test_embedding_generator.py

import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Add src to Python path
import sys
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root_dir)

from src.core.embed import EmbeddingGenerator

# A very small, fast model for testing if available, or skip if not.
# 'sentence-transformers/all-MiniLM-L6-v2' is relatively small.
TEST_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' 

# Fixture to manage model availability for tests
@pytest.fixture(scope="module")
def embedder_instance():
    try:
        instance = EmbeddingGenerator(model_name=TEST_MODEL_NAME)
        if instance.model is None: # Handle case where init sets model to None on failure
            pytest.skip(f"Skipping embedding tests: Model {TEST_MODEL_NAME} failed to load in fixture.")
        return instance
    except Exception as e:
        pytest.skip(f"Skipping embedding tests: Model {TEST_MODEL_NAME} raised exception during fixture load: {e}")

def test_embedding_generator_initialization(embedder_instance): # Uses fixture
    assert embedder_instance.model is not None

def test_embedding_generator_generate_single_embedding(embedder_instance): # Uses fixture
    text_chunks = ["This is a test sentence."]
    embeddings = embedder_instance.generate_embeddings(text_chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0 
    assert embeddings.dtype == np.float32

def test_embedding_generator_generate_multiple_embeddings(embedder_instance): # Uses fixture
    text_chunks = ["First sentence.", "Second sentence for testing."]
    embeddings = embedder_instance.generate_embeddings(text_chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float32

def test_embedding_generator_empty_input_list(embedder_instance): # Uses fixture
    text_chunks = []
    embeddings = embedder_instance.generate_embeddings(text_chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0 # Check for empty array more robustly

def test_embedding_generator_invalid_model_name_init(capsys):
    # Test initialization with a clearly invalid model name
    embedder = EmbeddingGenerator(model_name="this_is_definitely_not_a_real_model_name_ever_123")
    assert embedder.model is None
    captured = capsys.readouterr()
    assert "Error loading Sentence Transformer model" in captured.out
    assert "this_is_definitely_not_a_real_model_name_ever_123" in captured.out

    # Test generate_embeddings when model is None
    embeddings = embedder.generate_embeddings(["test"])
    assert embeddings is None
    captured = capsys.readouterr() # Capture new output
    assert "Embedding model not loaded. Cannot generate embeddings." in captured.out


# Test the case where model is initially loaded but then set to None (e.g. for some internal error)
# This is less about __init__ and more about generate_embeddings' robustness.
def test_generate_embeddings_with_model_set_to_none_post_init(embedder_instance, capsys):
    # This test assumes embedder_instance has a valid model initially.
    # We temporarily sabotage it.
    original_model = embedder_instance.model
    embedder_instance.model = None # Simulate model becoming unavailable
    
    embeddings = embedder_instance.generate_embeddings(["some text"])
    assert embeddings is None
    captured = capsys.readouterr()
    assert "Embedding model not loaded. Cannot generate embeddings." in captured.out
    
    embedder_instance.model = original_model # Restore for other tests if fixture is function-scoped
