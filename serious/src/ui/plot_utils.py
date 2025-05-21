# In src/ui/app.py or a new plot_utils.py
from pyvis.network import Network
import igraph as ig
import numpy as np
from typing import List, Optional, Tuple # Added Tuple
from typing import Dict, Any # Added for new function
import pandas as pd # Added for new function
import seaborn as sns # Added for new function
import matplotlib.pyplot as plt # Added for new function
from sklearn.feature_extraction.text import TfidfVectorizer # Added for new function
import os # For saving temp file
from scipy.sparse import csr_matrix # Added for type hint

def create_interactive_community_graph(
    embeddings: np.ndarray, # For node positions (optional, can use UMAP/tSNE)
    adj_matrix_sparse: csr_matrix, # The sparse adjacency matrix from kneighbors_graph
    communities: ig.VertexClustering,
    chunks: List[str], # To get text for node tooltips
    chunk_to_pdf_map: List[str], # Add this parameter
    output_filename: str = "community_graph.html",
    max_nodes_to_display: Optional[int] = 200
):
    if communities is None or not communities.membership:
        return None

    num_total_nodes = len(communities.membership)
    if num_total_nodes == 0:
        return None

    # 1. Calculate degrees for all nodes based on the undirected graph pyvis will render
    actual_degrees = [0] * num_total_nodes
    processed_edges: set[Tuple[int, int]] = set()
    
    s_nodes, t_nodes = adj_matrix_sparse.nonzero()
    for s_orig, t_orig in zip(s_nodes, t_nodes):
        s_int, t_int = int(s_orig), int(t_orig)
        if s_int != t_int: # Ignore self-loops
            u, v = min(s_int, t_int), max(s_int, t_int)
            if u < num_total_nodes and v < num_total_nodes: # Ensure indices are valid for communities/chunks
                processed_edges.add((u, v))

    for u, v in processed_edges:
        actual_degrees[u] += 1
        actual_degrees[v] += 1

    # 2. Determine sampled_node_original_indices
    sampled_node_original_indices: List[int]
    if max_nodes_to_display is not None and num_total_nodes > max_nodes_to_display:
        sorted_original_indices = sorted(
            range(num_total_nodes),
            key=lambda i: actual_degrees[i],
            reverse=True
        )
        sampled_node_original_indices = sorted_original_indices[:max_nodes_to_display]
    else:
        sampled_node_original_indices = list(range(num_total_nodes))

    if not sampled_node_original_indices:
        return None # No nodes to plot

    # Sort for consistent pyvis_id assignment if the same set of nodes is chosen
    sampled_node_original_indices.sort() 
    
    original_to_pyvis_id = {
        orig_idx: new_idx for new_idx, orig_idx in enumerate(sampled_node_original_indices)
    }

    # Create a pyvis network
    net = Network(notebook=True, height="750px", width="100%", cdn_resources='remote', directed=False)

    # Add sampled nodes
    node_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFA1']
    for pyvis_idx, original_idx in enumerate(sampled_node_original_indices):
        if original_idx >= len(communities.membership) or original_idx >= len(chunks) or original_idx >= len(chunk_to_pdf_map):
            # This should ideally not happen if inputs are consistent and checks above are robust
            continue

        membership_id = communities.membership[original_idx]
        chunk_text = chunks[original_idx]
        chunk_preview = chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text
        # Get PDF name
        pdf_name_for_node = chunk_to_pdf_map[original_idx]

        net.add_node(
            pyvis_idx,  # Use the new, contiguous pyvis_id
            label=f"Chunk {original_idx}", # Label refers to original identifier
            title=(
                f"Original Index: {original_idx}\n"
                f"Community: {membership_id}\n"
                f"Degree: {actual_degrees[original_idx]}\n"
                f"Source PDF: {pdf_name_for_node}\n" # Add source PDF to tooltip
                f"Text: {chunk_preview}"
            ),
            color=node_colors[membership_id % len(node_colors)],
            group=int(membership_id)
        )

    # Add edges connecting the sampled nodes
    for s_orig, t_orig in processed_edges:
        if s_orig in original_to_pyvis_id and t_orig in original_to_pyvis_id:
            pyvis_s = original_to_pyvis_id[s_orig]
            pyvis_t = original_to_pyvis_id[t_orig]
            net.add_edge(pyvis_s, pyvis_t)

    # Configure physics for better layout initially
    if net.nodes: # Only apply physics if there are nodes
        net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4)
    # net.show_buttons(filter_=['physics']) # Can add buttons for user to toggle physics

    # Ensure the output directory exists (e.g., a 'temp' folder in your project root)
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    full_path = os.path.join(temp_dir, output_filename)
    
    try:
        net.save_graph(full_path)
        return full_path
    except Exception:
        if not net.nodes: # Common case for error if pyvis can't save an empty graph
            return None 
        raise # Re-raise other unexpected errors

def plot_term_correlation_heatmap(chunks_by_community: Dict[int, List[str]], top_n_terms: int = 20):
    """
    Generates a heatmap showing the correlation between top terms across different communities
    based on their TF-IDF scores within each community's aggregated text.

    Args:
        chunks_by_community: A dictionary where keys are community IDs (int) and values
                             are lists of text chunks belonging to that community.
        top_n_terms: The number of top terms (based on overall TF-IDF across communities)
                     to include in the correlation analysis.

    Returns:
        A matplotlib Figure object containing the heatmap, or None if no data or terms
        are available for plotting.
    """
    if not chunks_by_community:
        return None

    community_corpus = [" ".join(texts) for cid, texts in chunks_by_community.items()]
    if not any(community_corpus): # All communities were empty
        return None
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n_terms, ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(community_corpus)
        terms = vectorizer.get_feature_names_out()
        
        # Create a term-community DataFrame
        df_term_community = pd.DataFrame(tfidf_matrix.toarray(), columns=terms, index=list(chunks_by_community.keys()))
        
        # Calculate term-term correlation across communities
        term_correlation_matrix = df_term_community.corr() # Term-wise correlation

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(term_correlation_matrix, annot=False, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        ax.set_title(f"Top {top_n_terms} Term Correlation Heatmap (Across Communities)")
        plt.tight_layout()
        return fig
    except ValueError as ve: # e.g. from TfidfVectorizer if corpus is empty after stop words
        print(f"Could not generate term correlation heatmap: {ve}") # Using print instead of st.warning
        return None

def plot_term_trends(dated_chunks: List[Dict[str, Any]], terms_to_track: List[str]):
    # dated_chunks: [{'text': "...", 'date': pd.Timestamp(...) }, ...]
    if not dated_chunks or not terms_to_track:
        return None

    df = pd.DataFrame(dated_chunks)
    # Note: st.warning/st.info are used in the original function.
    # If streamlit (st) is not available in this module's context,
    # these calls will fail. They are replaced with print statements below.
    if 'date' not in df.columns or 'text' not in df.columns:
        # st.warning("Dated chunks require 'date' and 'text' keys.")
        print("Warning: Dated chunks require 'date' and 'text' keys.")
        return None

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    for term in terms_to_track:
        # Use case-insensitive counting
        df[term] = df['text'].str.lower().str.count(term.lower())

    # Group by a time period (e.g., month) and sum counts
    df_trend = df.set_index('date').resample('ME')[terms_to_track].sum()

    if df_trend.empty or df_trend[terms_to_track].sum().sum() == 0: # Check if any terms had counts
        # st.info(f"No occurrences of tracked terms found over time.")
        print(f"Info: No occurrences of tracked terms found over time.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for term in terms_to_track:
        ax.plot(df_trend.index, df_trend[term], label=term, marker='o', linestyle='-')

    ax.set_title("Term Frequency Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Frequency Count (per month)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
