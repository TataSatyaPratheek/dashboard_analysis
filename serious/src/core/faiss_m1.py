# src/core/faiss_m1.py

import faiss
import numpy as np

class FAISSIndexer:
    def __init__(self, dim=384, m=8, nbits=8):
        """
        Initializes the FAISSIndexer with Product Quantization (PQ).
        Args:
            dim (int): Dimensionality of the embeddings.
            m (int): Number of sub-quantizers for PQ.
            nbits (int): Number of bits per sub-quantizer index.
        """
        # For M1, faiss-cpu will use optimized BLAS libraries if installed correctly (e.g., OpenBLAS)
        self.index = faiss.IndexPQ(dim, m, nbits)
        self.dim = dim
        self.is_trained = False
        
    def build(self, embeddings: np.ndarray):
        """
        Builds the FAISS index from the given embeddings.
        M1-optimized considerations include using float32 and C-contiguous arrays.
        Args:
            embeddings (np.ndarray): A NumPy array of embeddings.
        """
        if embeddings.shape[0] == 0:
            print("No embeddings provided to build the index.")
            return

        # Convert to contiguous array of type float32 for FAISS and M1 SIMD efficiency
        data = np.ascontiguousarray(embeddings, dtype='float32')
        
        if not self.index.is_trained:
            if data.shape[0] < self.index.pq.M * 256 : # Heuristic for sufficient training data for PQ
                 print(f"Warning: Low number of training vectors ({data.shape[0]}) for PQ. Index quality might be suboptimal.")
            self.index.train(data)
            self.is_trained = True

        self.index.add(data)
        
    def search(self, query_embeddings: np.ndarray, k=10):
        """
        Searches the index for the k-nearest neighbors of the query embeddings.
        Args:
            query_embeddings (np.ndarray): A NumPy array of query embeddings.
            k (int): The number of nearest neighbors to retrieve.
        Returns:
            tuple: Distances and indices of the k-nearest neighbors.
        """
        if not self.index.ntotal > 0:
            print("Index is empty. Cannot perform search.")
            return np.array([]), np.array([])
            
        query_data = np.ascontiguousarray(query_embeddings, dtype='float32')
        if query_data.ndim == 1: # Handle single query vector
            query_data = query_data.reshape(1, -1)

        return self.index.search(query_data, k)

