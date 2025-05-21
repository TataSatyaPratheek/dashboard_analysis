# src/core/faiss_m1.py

import faiss
import numpy as np

class FAISSIndexer:
    def __init__(self, dim=384, m=4, nbits=4):
        """
        Initializes the FAISSIndexer with Product Quantization (PQ).
        Args:
            dim (int): Dimensionality of the embeddings.
            m (int): Number of sub-quantizers for PQ.
            nbits (int): Number of bits per sub-quantizer index.
        """
        self.dim = dim
        self.pq_m = m  # Store PQ params for reset
        self.pq_nbits = nbits # Store PQ params for reset
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
        
        if not self.is_trained: # Check internal trained flag
            # Heuristic for sufficient training data for PQ
            # For IndexPQ, typical rule of thumb is at least k * 256 vectors for training, where k is pq.M.
            # Or more generally 30*M to 200*M values.
            # Faiss docs for IndexPQ: "The training vectors should be representative of the data to be indexed."
            # If data.shape[0] is very small compared to what PQ needs, training might be poor.
            # A common recommendation is at least 1000s of vectors for PQ training if possible.
            # The existing check `data.shape[0] < self.index.pq.M * 256` is a reasonable heuristic.
            if data.shape[0] < self.index.pq.M * 256 and data.shape[0] > 0 : 
                print(f"Warning: Low number of training vectors ({data.shape[0]}) for PQ (m={self.index.pq.M}). Index quality might be suboptimal.")
            
            if data.shape[0] > 0: # Ensure there's data to train on
                self.index.train(data)
                self.is_trained = True # Set internal flag
            else:
                print("No data to train the index.")
                return # Cannot add if not trained and no data to train

        if self.is_trained: # Only add if trained
            self.index.add(data)
        else:
            print("Index is not trained and could not be trained (e.g. no data). Cannot add embeddings.")

    def search(self, query_embeddings: np.ndarray, k=10):
        """
        Args:
            query_embeddings (np.ndarray): A NumPy array of query embeddings.
            k (int): The number of nearest neighbors to retrieve.
        Returns:
            tuple: Distances and indices of the k-nearest neighbors.
        """
        if not self.is_trained or not self.index.ntotal > 0: # Check if trained and has data
            return np.array([]), np.array([])
            
        query_data = np.ascontiguousarray(query_embeddings, dtype='float32')
        if query_data.ndim == 1: # Handle single query vector
            query_data = query_data.reshape(1, -1)

        return self.index.search(query_data, k)

    def reset(self):
        """Resets the index to an untrained state."""
        print("Resetting FAISS index.")
        if self.index:
            self.index.reset() # This clears data and trained state for most Faiss indexes
        # Re-initialize the IndexPQ object to be absolutely sure
        self.index = faiss.IndexPQ(self.dim, self.pq_m, self.pq_nbits)
        self.is_trained = False
