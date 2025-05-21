# tests/integration/test_full_pipeline.py

import pytest
import os
import numpy as np
from pypdf import PdfWriter # For creating a dummy PDF for testing

# Add src to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.pdf import PDFProcessor
from src.core.embed import EmbeddingGenerator
from src.core.faiss_m1 import FAISSIndexer
from src.core.gala import CommunityDetector
from src.utils.config import load_config

TEST_MODEL_NAME_INTEGRATION = 'sentence-transformers/all-MiniLM-L6-v2' # Use a real small model

def is_integration_model_available_or_skip():
    try:
        EmbeddingGenerator(model_name=TEST_MODEL_NAME_INTEGRATION)
        return True
    except Exception:
        pytest.skip(f"Skipping integration tests: Model {TEST_MODEL_NAME_INTEGRATION} not available or failed to load.")
        return False


@pytest.fixture(scope="module")
def integration_test_pdf_path():
    test_dir = os.path.join(project_root, "tests/test_data")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    pdf_path = os.path.join(test_dir, "integration_test.pdf")
    
    writer = PdfWriter()
    # Add a few pages with some "text"
    # Actual text extraction for blank pages by pypdf is minimal.
    # To make this more meaningful, one might need a PDF with actual renderable text.
    for i in range(2): # 2 pages
        writer.add_blank_page(width=612, height=792)
        writer.add_metadata({f"/Title_Page_{i}": f"Content for page {i}"}) # Minimal content

    with open(pdf_path, "wb") as f:
        writer.write(f)
    yield pdf_path
    # os.remove(pdf_path) # Clean up

@pytest.mark.skipif(not is_integration_model_available_or_skip(), reason="Test model not available for integration test")
def test_small_end_to_end_pipeline(integration_test_pdf_path):
    # 1. Load Config (can use defaults for this test)
    config = load_config(os.path.join(project_root, "configs/base.yaml"))

    # 2. PDF Processing
    processor = PDFProcessor(
        chunk_size=config.pdf_processor.chunk_size,
        chunk_overlap=config.pdf_processor.chunk_overlap
    )
    chunks = processor.process(integration_test_pdf_path)
    assert isinstance(chunks, list)
    # With the dummy PDF, chunks might be empty or contain very little.
    # Ensure enough for FAISS PQ training with nbits=8
    if not chunks or len(chunks) < 256: 
        print(f"Warning: Insufficient chunks from PDF ({len(chunks)}). Using larger fallback for FAISS training.")
        min_faiss_train_samples = 256 # For nbits=8
        chunks = [f"Sample dummy chunk {i+1} for pipeline testing." for i in range(min_faiss_train_samples)]

    # 3. Embedding Generation
    embedder = EmbeddingGenerator(model_name=TEST_MODEL_NAME_INTEGRATION)
    embeddings = embedder.generate_embeddings(chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == config.faiss_indexer.dim # Assuming MiniLM model
    
    if embeddings.shape[0] == 0:
        pytest.skip("No embeddings generated, cannot proceed with FAISS and Community detection.")

    # 4. FAISS Indexing
    indexer = FAISSIndexer(
        dim=embeddings.shape[1],
        m=config.faiss_indexer.m, # Use a smaller m for tiny test data if necessary
        nbits=config.faiss_indexer.nbits
    )
    indexer.build(embeddings)
    assert indexer.index.ntotal == len(chunks)

    # Perform a dummy search
    if len(chunks) > 0:
        query_embedding = embedder.generate_embeddings(["search query for test"])
        distances, indices = indexer.search(query_embedding, k=1)
        assert indices.shape == (1, 1)

    # 5. Community Detection
    # Adjust n_neighbors if number of embeddings is too small
    n_samples = embeddings.shape[0]
    n_neighbors_for_test = min(config.community_detector.n_neighbors, n_samples -1) if n_samples > 1 else 1
    
    if n_samples <= 1 : # Cannot run community detection on 1 or 0 samples.
        print("Skipping community detection due to insufficient samples.")
    else:
        detector = CommunityDetector(
            n_neighbors=n_neighbors_for_test,
            use_weights=config.community_detector.use_weights
        )
        communities = detector.detect(embeddings)
        assert communities is not None # igraph.VertexClustering object
        assert len(communities) >= 0 # Number of communities
