# src/ui/__init__.py

"""
UI package for the dashboard application.

This package contains the Streamlit application (app.py) and
utility functions for plotting (plot_utils.py).
"""

from .plot_utils import (
    create_interactive_community_graph,
    plot_term_correlation_heatmap,
    plot_term_trends,
)

__all__ = [
    "create_interactive_community_graph",
    "plot_term_correlation_heatmap",
    "plot_term_trends",
]