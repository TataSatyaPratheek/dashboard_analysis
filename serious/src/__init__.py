# src/__init__.py

"""
Main application package for the Dashboard Analysis tool.

This package aggregates core functionalities, LLM integrations, UI components,
and utility functions, making them easily accessible.
"""

# Import from core sub-package
from .core import (
    PDFProcessor,
    EmbeddingGenerator,
    FAISSIndexer,
    CommunityDetector
)

# Import from llm sub-package
from .llm import (
    OpenAIQuestionGenerator,
    OllamaQuestionGenerator
)

# Import from ui sub-package (specifically plot_utils)
# The main Streamlit app (app.py) is typically run as a script
# and not imported as a module component directly.
from .ui import (
    create_interactive_community_graph,
    plot_term_correlation_heatmap,
    plot_term_trends
)

# Import from utils sub-package
from .utils import (
    load_config,
    ensure_contiguous_float32
)

__all__ = [
    # Core components
    "PDFProcessor", "EmbeddingGenerator", "FAISSIndexer", "CommunityDetector",

    # LLM components
    "OpenAIQuestionGenerator", "OllamaQuestionGenerator",

    # UI plot utilities
    "create_interactive_community_graph", "plot_term_correlation_heatmap", "plot_term_trends",

    # Utility functions
    "load_config", "ensure_contiguous_float32",
]