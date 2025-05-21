# src/core/gala.py

import igraph as ig
import numpy as np
import scipy.sparse # For checking sparse matrix type
from sklearn.neighbors import NearestNeighbors # For KNN graph construction

class CommunityDetector:
    def __init__(self, n_neighbors=15, use_weights=False):
        """
        Initializes the CommunityDetector.
        Args:
            n_neighbors (int): Number of neighbors for KNN graph construction.
            use_weights (bool): Whether to use distances as weights in the graph.
                                Louvain can use weights.
        """
        self.n_neighbors = n_neighbors
        self.use_weights = use_weights
        
    def detect(self, embeddings: np.ndarray):
        """
        Detects communities in the data using the Louvain algorithm on a KNN graph.
        CPU-optimized by leveraging efficient libraries like scikit-learn and igraph.
        Args:
            embeddings (np.ndarray): A NumPy array of embeddings.
        Returns:
            igraph.VertexClustering: An object representing the detected communities.
                                     Returns None if embeddings are insufficient.
        """
        if embeddings.shape[0] < self.n_neighbors:
            print(f"Not enough embeddings ({embeddings.shape[0]}) to build a KNN graph with {self.n_neighbors} neighbors.")
            return None

        # 1. Build KNN graph
        # Ensure embeddings are float32 for consistency and potential SIMD optimizations
        data = np.ascontiguousarray(embeddings, dtype='float32')
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto', metric='euclidean', n_jobs=-1).fit(data)
        adj_matrix = nbrs.kneighbors_graph(data, mode='distance' if self.use_weights else 'connectivity')
        
        # 2. Convert to igraph Graph
        # Ensure adj_matrix is not an np.matrix before calling .nonzero(),
        # as np.matrix is deprecated and its .nonzero() can have unexpected behavior
        # (e.g., returning an empty tuple instead of two empty arrays).
        # sklearn's kneighbors_graph typically returns a scipy.sparse matrix,
        # whose .nonzero() is fine. This handles cases where it might be np.matrix (e.g., in tests).
        if isinstance(adj_matrix, np.matrix):
            print("DEBUG: adj_matrix was np.matrix, converting to ndarray")
            adj_matrix = np.asarray(adj_matrix)

        print(f"DEBUG: adj_matrix type before nonzero: {type(adj_matrix)}")
        print(f"DEBUG: adj_matrix shape: {adj_matrix.shape}")
        print(f"DEBUG: adj_matrix content (first few elements or all if small):\n{adj_matrix[:5,:5] if adj_matrix.size > 0 else adj_matrix}")
        
        if scipy.sparse.issparse(adj_matrix):
            print(f"DEBUG: Number of non-zero elements in adj_matrix: {adj_matrix.nnz}")
        else: # Handles np.ndarray (and np.matrix after conversion)
            print(f"DEBUG: Number of non-zero elements in adj_matrix: {np.count_nonzero(adj_matrix)}")

        #    For weighted Louvain, pass the weights.
        sources, targets = adj_matrix.nonzero()

        if self.use_weights:
            weights_values = adj_matrix[sources, targets]
            # Ensure weights_values is a flat list of Python floats/ints
            if hasattr(weights_values, 'A1'): # Handles np.matrix
                weights_list = weights_values.A1.tolist()
            elif hasattr(weights_values, 'toarray'): # Handles sparse matrices if somehow [sources,targets] returns one
                weights_list = weights_values.toarray().flatten().tolist()
            elif isinstance(weights_values, np.ndarray):
                weights_list = weights_values.flatten().tolist()
            else:
                weights_list = list(weights_values) # Fallback for other iterables

            g = ig.Graph(list(zip(sources, targets)), directed=False, edge_attrs={'weight': weights_list})
        else:
            g = ig.Graph(list(zip(sources, targets)), directed=False)

        # 3. Memory-efficient Louvain community detection from igraph
        #    The 'community_multilevel' method in python-igraph is an efficient Louvain implementation.
        #    If using weights, it will consider them.
        if self.use_weights:
            communities = g.community_multilevel(weights="weight", return_levels=False) # More idiomatic to pass attribute name
        else:
            communities = g.community_multilevel(return_levels=False)
            
        return communities
