# scripts/benchmark.py

import time
import numpy as np
import os
import sys

# Add src to Python path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.core.pdf import PDFProcessor
from src.core.embed import EmbeddingGenerator
from src.core.faiss_m1 import FAISSIndexer
from src.core.gala import CommunityDetector
from src.utils.config import load_config # Use the updated load_config

# --- Helper to create a dummy PDF if one doesn't exist ---
def ensure_dummy_pdf(filepath="data/sample_benchmark.pdf"):
    if not os.path.exists(filepath):
        print(f"Warning: Benchmark PDF '{filepath}' not found. Skipping PDF processing benchmark.")
        return False
    # Minimal check, actual content doesn't matter much for this basic benchmark
    if os.path.getsize(filepath) == 0:
        print(f"Warning: Benchmark PDF '{filepath}' is empty. Skipping PDF processing benchmark.")
        return False
    return True

def run_benchmarks():
    print("Loading configurations for benchmark...")
    # Load base and M1-specific configurations
    config = load_config(os.path.join(project_root, "configs/base.yaml"), 
                         os.path.join(project_root, "configs/m1_optim.yaml"))

    print(f"Using embedding dimension: {config.faiss_indexer.dim}")
    print(f"Test vectors: {config.benchmark.num_test_vectors}, Query vectors: {config.benchmark.num_query_vectors}")

    # --- 0. Prepare Data ---
    pdf_file_path = os.path.join(project_root, config.benchmark.pdf_test_file)
    dummy_embeddings = np.random.rand(config.benchmark.num_test_vectors, config.faiss_indexer.dim).astype('float32')
    dummy_queries = np.random.rand(config.benchmark.num_query_vectors, config.faiss_indexer.dim).astype('float32')

    results = {}

    # --- 1. PDF Processing Benchmark ---
    if ensure_dummy_pdf(pdf_file_path):
        print("\nBenchmarking PDF Processing...")
        processor = PDFProcessor(
            chunk_size=config.pdf_processor.chunk_size,
            chunk_overlap=config.pdf_processor.chunk_overlap
        )
        start_time = time.perf_counter()
        chunks = processor.process(pdf_file_path)
        end_time = time.perf_counter()
        results['pdf_processing_time_s'] = end_time - start_time
        results['pdf_num_chunks'] = len(chunks)
        print(f"PDF Processing: {results['pdf_processing_time_s']:.4f} s, Chunks: {len(chunks)}")
        # Use actual chunks for embedding if available and not too many
        if chunks and len(chunks) < 2000: # Limit to avoid excessive embedding time for benchmark
            sample_chunks_for_embedding = chunks
        else:
            sample_chunks_for_embedding = [f"Sample text chunk {i}" for i in range(100)] # Fallback
    else:
        print("Skipping PDF Processing benchmark as test PDF is missing or invalid.")
        sample_chunks_for_embedding = [f"Sample text chunk {i}" for i in range(100)] # Fallback


    # --- 2. Embedding Generation Benchmark ---
    print("\nBenchmarking Embedding Generation...")
    embedder = EmbeddingGenerator(model_name=config.embedding_generator.model_name)
    if embedder.model: # Check if model loaded
        start_time = time.perf_counter()
        # Embed either the processed chunks (if few) or a fixed number of dummy chunks
        actual_embeddings = embedder.generate_embeddings(sample_chunks_for_embedding)
        end_time = time.perf_counter()
        results['embedding_generation_time_s'] = end_time - start_time
        results['embedding_num_vectors'] = actual_embeddings.shape[0] if actual_embeddings is not None else 0
        print(f"Embedding Generation ({results.get('embedding_num_vectors', 0)} vectors): {results.get('embedding_generation_time_s', 0):.4f} s")
        
        # Use generated embeddings for next steps if available, else use dummy
        if actual_embeddings is not None and actual_embeddings.shape[0] > 0:
            benchmark_embeddings = np.ascontiguousarray(actual_embeddings, dtype='float32')
            if benchmark_embeddings.shape[1] != config.faiss_indexer.dim:
                 print(f"Warning: Actual embedding dim ({benchmark_embeddings.shape[1]}) != config dim ({config.faiss_indexer.dim}). Using dummy embeddings for FAISS.")
                 benchmark_embeddings = dummy_embeddings # Fallback if dims mismatch
        else:
            benchmark_embeddings = dummy_embeddings
    else:
        print("Embedding model failed to load. Using dummy embeddings for subsequent benchmarks.")
        benchmark_embeddings = dummy_embeddings
        results['embedding_generation_time_s'] = -1 # Indicate failure

    # --- 3. FAISS Indexing Benchmark ---
    print("\nBenchmarking FAISS Indexing...")
    indexer = FAISSIndexer(
        dim=benchmark_embeddings.shape[1], # Use actual dim from embeddings
        m=config.faiss_indexer.m,
        nbits=config.faiss_indexer.nbits
    )
    # Build Time
    start_time = time.perf_counter()
    indexer.build(benchmark_embeddings)
    end_time = time.perf_counter()
    results['faiss_build_time_s'] = end_time - start_time
    print(f"FAISS Index Build: {results['faiss_build_time_s']:.4f} s")

    # Search Time (Average over queries)
    # Ensure query embeddings match index dimension
    if dummy_queries.shape[1] != benchmark_embeddings.shape[1]:
        print(f"Warning: Query dim ({dummy_queries.shape[1]}) != index dim ({benchmark_embeddings.shape[1]}). Re-generating dummy queries.")
        current_dim = benchmark_embeddings.shape[1]
        dummy_queries_for_search = np.random.rand(config.benchmark.num_query_vectors, current_dim).astype('float32')
    else:
        dummy_queries_for_search = dummy_queries

    search_times = []
    if indexer.index.ntotal > 0:
        for i in range(dummy_queries_for_search.shape[0]):
            query_vec = dummy_queries_for_search[i:i+1, :]
            start_time = time.perf_counter()
            _, _ = indexer.search(query_vec, k=config.faiss_indexer.k_search)
            end_time = time.perf_counter()
            search_times.append(end_time - start_time)
        results['faiss_avg_query_time_ms'] = (sum(search_times) / len(search_times)) * 1000 if search_times else 0
        print(f"FAISS Avg Query Time: {results['faiss_avg_query_time_ms']:.4f} ms")
    else:
        print("FAISS index is empty, skipping search benchmark.")
        results['faiss_avg_query_time_ms'] = -1


    # --- 4. Community Detection Benchmark ---
    print("\nBenchmarking Community Detection...")
    if benchmark_embeddings.shape[0] > config.community_detector.n_neighbors :
        detector = CommunityDetector(
            n_neighbors=config.community_detector.n_neighbors,
            use_weights=config.community_detector.use_weights
        )
        start_time = time.perf_counter()
        communities = detector.detect(benchmark_embeddings)
        end_time = time.perf_counter()
        results['community_detection_time_s'] = end_time - start_time
        results['num_communities_detected'] = len(communities) if communities else 0
        print(f"Community Detection: {results['community_detection_time_s']:.4f} s, Communities: {results.get('num_communities_detected', 0)}")
    else:
        print(f"Not enough embeddings ({benchmark_embeddings.shape[0]}) for community detection with n_neighbors={config.community_detector.n_neighbors}. Skipping.")
        results['community_detection_time_s'] = -1


    # --- Print Summary ---
    print("\n--- Benchmark Summary ---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Create dummy data directory and a tiny PDF if they don't exist
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    dummy_pdf_path = os.path.join(project_root, "data/sample_benchmark.pdf")
    if not os.path.exists(dummy_pdf_path):
        try:
            from pypdf import PdfWriter # Using pypdf, ensure it's installed
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792) # Standard US Letter
            with open(dummy_pdf_path, "wb") as f:
                writer.write(f)
            print(f"Created dummy PDF: {dummy_pdf_path}")
        except ImportError:
            print("pypdf not installed. Cannot create dummy PDF. Please create 'data/sample_benchmark.pdf' manually.")
        except Exception as e:
            print(f"Could not create dummy PDF: {e}")
            
    run_benchmarks()
