# src/ui/app.py

import streamlit as st
import numpy as np
import time # To simulate processing time if needed

# Assuming the project structure where 'src' is in PYTHONPATH
# or the app is run from the project root.
from src.core.pdf import PDFProcessor
from src.core.embed import EmbeddingGenerator
from src.core.faiss_m1 import FAISSIndexer
from src.core.gala import CommunityDetector
from src.utils.config import load_config # Load configuration

# --- Helper Functions for UI ---
def display_communities_info(communities, chunks):
    if communities is None:
        st.warning("Community detection could not be performed (e.g., too few items).")
        return
        
    st.subheader(f"Detected {len(communities)} Insight Communities")
    if len(communities) == 0:
        st.info("No distinct communities were detected with the current settings.")
        return

    for i, community_indices in enumerate(communities):
        with st.expander(f"Community {i+1} ({len(community_indices)} items)"):
            # Display first few items from the community
            for item_index in community_indices[:min(5, len(community_indices))]: # Show up to 5 items
                st.caption(f"- Chunk {item_index}: {chunks[item_index][:150]}...") # Display snippet

def visualize_graph_placeholder(embeddings, communities):
    # Placeholder for actual graph visualization (e.g., using Pyvis, Plotly, or Streamlit's graph_chart)
    # This would typically involve creating a graph from embeddings and community assignments
    if communities and embeddings.shape[0] > 0:
        st.write(f"Graph Visualization (Placeholder): {len(embeddings)} nodes, {len(communities)} communities.")
        # Example: st.graph_chart could be used if data is formatted correctly for it.
        # For more complex graphs, you might save an HTML file (e.g. from Pyvis) and embed it.
    else:
        st.info("Not enough data to visualize graph.")

# --- Main Application Logic ---
def main():
    # Load configurations
    app_config = load_config() # Loads from configs/base.yaml by default

    st.set_page_config(page_title=app_config.streamlit_ui['title'], layout="wide")
    st.title(app_config.streamlit_ui['title'])

    # Initialize core components based on config
    # These are initialized once and reused or re-initialized if config changes
    # For simplicity, we initialize them on first use or after file upload here.
    
    pdf_file = st.file_uploader("Upload SEO Report (PDF)", type="pdf")

    if pdf_file:
        st.info(f"Processing '{pdf_file.name}'...")
        start_time = time.time()

        # Core processing pipeline
        with st.spinner("Step 1: Extracting and chunking text from PDF..."):
            processor = PDFProcessor(
                chunk_size=app_config.pdf_processor['chunk_size'],
                chunk_overlap=app_config.pdf_processor['chunk_overlap']
            )
            # Pass the file buffer directly to PDFProcessor
            chunks = processor.process(pdf_file) 
        st.success(f"Extracted {len(chunks)} text chunks.")

        if not chunks:
            st.error("No text could be extracted from the PDF. Please check the file.")
            return

        with st.spinner("Step 2: Generating embeddings for text chunks..."):
            embedder = EmbeddingGenerator(model_name=app_config.embedding_generator['model_name'])
            if embedder.model is None: # Check if model loaded successfully
                st.error("Failed to load embedding model. Cannot proceed.")
                return
            embeddings = embedder.generate_embeddings(chunks)
        st.success(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
        
        if embeddings.shape[0] == 0:
            st.error("No embeddings were generated. Cannot proceed.")
            return

        with st.spinner("Step 3: Building FAISS index..."):
            indexer = FAISSIndexer(
                dim=embeddings.shape[1], # Use actual embedding dimension
                m=app_config.faiss_indexer['m'],
                nbits=app_config.faiss_indexer['nbits']
            )
            indexer.build(embeddings)
        st.success("FAISS index built.")

        with st.spinner("Step 4: Detecting communities using GALA-Louvain (CPU)..."):
            detector = CommunityDetector(
                n_neighbors=min(app_config.community_detector['n_neighbors'], embeddings.shape[0]-1) if embeddings.shape[0] > 1 else 1, # Ensure n_neighbors < n_samples
                use_weights=app_config.community_detector['use_weights']
            )
            communities = detector.detect(embeddings)
        st.success("Community detection complete.")
        
        processing_time = time.time() - start_time
        st.info(f"Total processing time: {processing_time:.2f} seconds.")

        # Display Results
        display_communities_info(communities, chunks)
        
        # Placeholder for interactive search or further analysis
        st.subheader("Explore Embeddings (Placeholder)")
        if embeddings.shape[0] > 0:
            query_text = st.text_input("Search for similar content (experimental):")
            if query_text:
                query_embedding = embedder.generate_embeddings([query_text])
                if query_embedding is not None and query_embedding.shape[0] > 0 :
                    distances, indices = indexer.search(query_embedding, k=app_config.faiss_indexer['k_search'])
                    st.write("Search Results (Top K):")
                    for i, idx in enumerate(indices[0]):
                        st.caption(f"Rank {i+1} (Dist: {distances[0][i]:.4f}): Chunk {idx} - {chunks[idx][:150]}...")
                else:
                    st.warning("Could not generate embedding for the query.")
        
        # Placeholder for graph visualization
        # visualize_graph_placeholder(embeddings, communities)

if __name__ == '__main__':
    # Note: To run this Streamlit app, you would typically use:
    # streamlit run src/ui/app.py
    # Ensure 'pypdf', 'langchain', 'sentence-transformers', 'faiss-cpu', 
    # 'scikit-learn', 'python-igraph', 'numpy', 'streamlit', 'pyyaml' are installed.
    main()
