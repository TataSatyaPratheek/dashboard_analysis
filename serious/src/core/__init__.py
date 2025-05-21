# src/core/__init__.py

# This file makes the 'core' directory a Python package.
# You can optionally import key classes or functions here
# to make them available directly from the core package.

from .pdf import PDFProcessor
from .embed import EmbeddingGenerator
from .faiss_m1 import FAISSIndexer
from .gala import CommunityDetector

__all__ = [
    "PDFProcessor",
    "EmbeddingGenerator",
    "FAISSIndexer",
    "CommunityDetector"
]
