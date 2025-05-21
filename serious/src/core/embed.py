# src/core/embed.py

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the EmbeddingGenerator with a pre-trained Sentence Transformer model.
        Args:
            model_name (str): The name of the Sentence Transformer model to use.
                              'all-MiniLM-L6-v2' is a good default for performance and quality.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            # Handle model loading errors (e.g., network issues, invalid model name)
            print(f"Error loading Sentence Transformer model '{model_name}': {e}")
            self.model = None # Or raise the exception

    def generate_embeddings(self, text_chunks: list[str]) -> np.ndarray | None:
        """
        Generates embeddings for a list of text chunks.
        Args:
            text_chunks (list[str]): A list of text strings.
        Returns:
            np.ndarray | None: A NumPy array of embeddings, or None if model failed to load.
                               Each row corresponds to an embedding for a text chunk.
        """
        if not self.model:
            print("Embedding model not loaded. Cannot generate embeddings.")
            return None
        if not text_chunks:
            return np.array([])
            
        # M1-optimized by batch processing if sentence-transformers supports it well.
        # The library handles underlying optimizations (e.g., PyTorch MPS if configured).
        embeddings = self.model.encode(text_chunks, show_progress_bar=False) # Set show_progress_bar=True for long tasks
        return np.asarray(embeddings, dtype='float32')

