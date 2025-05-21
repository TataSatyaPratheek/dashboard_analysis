# In src/ui/app.py or a new plot_utils.py
from pyvis.network import Network
import igraph as ig
import numpy as np
from typing import List, Optional
from typing import Dict, Any # Added for new function
import pandas as pd # Added for new function
import seaborn as sns # Added for new function
import matplotlib.pyplot as plt # Added for new function
from sklearn.feature_extraction.text import TfidfVectorizer # Added for new function
import os # For saving temp file

def create_interactive_community_graph(
    embeddings: np.ndarray, # For node positions (optional, can use UMAP/tSNE)
    adj_matrix_sparse, # The sparse adjacency matrix from kneighbors_graph
    communities: ig.VertexClustering,
    chunks: List[str], # To get text for node tooltips
    output_filename: str = "community_graph.html"
):
    if communities is None or len(communities.membership) == 0:
        return None

    # Create a pyvis network
    net = Network(notebook=True, height="750px", width="100%", cdn_resources='remote', directed=False)

    # Add nodes
    node_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFA1']
    for i, membership_id in enumerate(communities.membership):
        chunk_preview = chunks[i][:50] + "..." if len(chunks[i]) > 50 else chunks[i]
        net.add_node(
            int(i),  # Ensure node ID is a standard Python int
            label=f"Chunk {i}",
            title=f"Community: {membership_id}\nText: {chunk_preview}", # Tooltip
            color=node_colors[membership_id % len(node_colors)],
            group=int(membership_id) # Ensure group ID is a standard Python int
        )

    # Add edges from the adjacency matrix (only if they connect nodes within a certain proximity)
    sources, targets = adj_matrix_sparse.nonzero()
    # Optional: Get weights if you want to vary edge thickness/color
    # weights = adj_matrix_sparse[sources, targets] if hasattr(adj_matrix_sparse, '__getitem__') else [1] * len(sources)

    for i in range(len(sources)):
        # Add edges if they are not self-loops (though kneighbors_graph shouldn't produce them unless k=0)
        if sources[i] != targets[i]:
            # Pyvis might re-add nodes if they don't exist, but we added them all.
            net.add_edge(int(sources[i]), int(targets[i])) # Ensure edge node IDs are standard Python ints

    # Configure physics for better layout initially
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4)
    # net.show_buttons(filter_=['physics']) # Can add buttons for user to toggle physics

    # Ensure the output directory exists (e.g., a 'temp' folder in your project root)
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    full_path = os.path.join(temp_dir, output_filename)
    
    net.save_graph(full_path)
    return full_path

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
