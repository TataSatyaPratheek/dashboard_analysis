# src/ui/app.py
import asyncio
import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.neighbors import NearestNeighbors

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src import (
    CommunityDetector,
    EmbeddingGenerator,
    FAISSIndexer,
    OllamaQuestionGenerator,
    OpenAIQuestionGenerator,
    PDFProcessor,
    create_interactive_community_graph,
    load_config,
    plot_term_correlation_heatmap,
    plot_term_trends
)

# --- UI Helper Imports ---
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
            for item_index in community_indices[:min(5, len(community_indices))]:
                st.caption(f"- Chunk {item_index}: {chunks[item_index][:150]}...")


def visualize_graph_placeholder(embeddings, communities):
    if communities and embeddings.shape[0] > 0:
        st.write(f"Graph Visualization (Placeholder): {len(embeddings)} nodes, {len(communities)} communities.")
    else:
        st.info("Not enough data to visualize graph.")


# --- Cached Component Initialization ---
@st.cache_resource
def initialize_pdf_processor(chunk_size, chunk_overlap):
    return PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@st.cache_resource
def initialize_embedding_generator(model_name):
    return EmbeddingGenerator(model_name=model_name)


@st.cache_resource
def initialize_faiss_indexer(dim, m, nbits):
    return FAISSIndexer(dim=dim, m=m, nbits=nbits)


@st.cache_resource
def initialize_community_detector(n_neighbors, use_weights):
    return CommunityDetector(n_neighbors=n_neighbors, use_weights=use_weights)


@st.cache_resource
def initialize_openai_question_generator(model, temperature, max_tokens):
    return OpenAIQuestionGenerator(model=model, temperature=temperature, max_tokens=max_tokens)


@st.cache_resource
def initialize_ollama_question_generator(model, host):
    return OllamaQuestionGenerator(model=model, host=host)


# --- Data Processing Functions with Caching ---
@st.cache_data
def process_pdf(_processor, pdf_file_bytes, filename_for_cache_key):
    pdf_stream = io.BytesIO(pdf_file_bytes)
    return _processor.process(pdf_stream)


@st.cache_data
def generate_embeddings(_embedder, chunks):
    return _embedder.generate_embeddings(chunks)


@st.cache_data
def build_faiss_index(_indexer, embeddings):
    _indexer.build(embeddings)
    return _indexer


@st.cache_data
def detect_communities(_detector, embeddings):
    return _detector.detect(embeddings)


@st.cache_data
def create_adjacency_matrix(embeddings, n_neighbors):
    if embeddings.shape[0] < 2 or n_neighbors <= 0: # Corrected from <= 1 to < 2 for clarity if shape[0] can be 0 or 1
        return None
    vis_nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm='auto', metric='euclidean', n_jobs=-1
    ).fit(np.ascontiguousarray(embeddings, dtype='float32'))
    return vis_nbrs.kneighbors_graph(
        np.ascontiguousarray(embeddings, dtype='float32'),
        mode='connectivity'
    )


@st.cache_data
def generate_questions_for_community(_q_generator, community_texts, community_id, num_questions):
    return _q_generator.generate_questions_from_community_texts(
        community_texts,
        community_id=community_id,
        num_questions=num_questions
    )


# --- Fragments ---
@st.fragment
def display_processing_section(app_config):
    st.subheader("PDF Processing")
    pdf_file_uploader = st.file_uploader("Upload SEO Report (PDF)", type="pdf", key="pdf_uploader")

    if pdf_file_uploader:
        pdf_file_contents = pdf_file_uploader.getvalue()
        # Only process if it's a new file or the first time a file is uploaded, or if processed_data is None
        if "pdf_file_name" not in st.session_state or \
           st.session_state.pdf_file_name != pdf_file_uploader.name or \
           st.session_state.processed_data is None:

            st.session_state.pdf_file_name = pdf_file_uploader.name
            st.info(f"Processing '{st.session_state.pdf_file_name}'...")
            start_time = time.time()
            # Ensure flag is false before starting a new processing job
            st.session_state.processing_complete_for_current_file = False

            # Initialize cached resources
            processor = initialize_pdf_processor(
                app_config.pdf_processor.chunk_size,
                app_config.pdf_processor.chunk_overlap
            )

            # Process PDF (cached) - passing bytes and filename for cache key
            chunks = process_pdf(processor, pdf_file_contents, st.session_state.pdf_file_name)

            if not chunks:
                st.error("No text could be extracted from the PDF. Please check the file.")
                st.session_state.processed_data = None
                return

            # Generate embeddings (cached)
            embedder = initialize_embedding_generator(app_config.embedding_generator.model_name)
            if embedder.model is None:
                st.error("Failed to load embedding model. Cannot proceed.")
                st.session_state.processed_data = None
                return

            embeddings = generate_embeddings(embedder, chunks)
            if embeddings.shape[0] == 0:
                st.error("No embeddings were generated. Cannot proceed.")
                st.session_state.processed_data = None
                return

            # Build FAISS index (cached)
            indexer_obj = initialize_faiss_indexer(
                dim=embeddings.shape[1],
                m=app_config.faiss_indexer.m,
                nbits=app_config.faiss_indexer.nbits
            )
            indexer_obj = build_faiss_index(indexer_obj, embeddings)

            # Detect communities (cached)
            n_neighbors_val = 1
            if embeddings.shape[0] > 1:
                n_neighbors_val = min(app_config.community_detector.n_neighbors, embeddings.shape[0] - 1)

            detector = initialize_community_detector(
                n_neighbors=n_neighbors_val,
                use_weights=app_config.community_detector.use_weights
            )
            communities = detect_communities(detector, embeddings)

            processing_time = time.time() - start_time

            # Store results in session state for other fragments
            st.session_state.processed_data = {
                "chunks": chunks,
                "embeddings": embeddings,
                "communities": communities,
                "indexer": indexer_obj,
                "embedder": embedder,
                "processing_time": processing_time,
                "pdf_file_name": st.session_state.pdf_file_name
            }
            st.success(f"PDF processed in {processing_time:.2f} seconds.")
            # Set a flag to indicate processing is complete for THIS UPLOAD
            st.session_state.processing_complete_for_current_file = True
            st.rerun()
        elif not st.session_state.get("processing_complete_for_current_file", False):
            # If the file is the same but processing wasn't marked complete (e.g., due to a previous error mid-process)
            # allow reprocessing by clearing the flag and rerunning (which will then hit the processing block)
            st.session_state.pdf_file_name = None # Force re-evaluation of the file
            st.rerun()
        # If file is same and was processed, do nothing to avoid loop, but show info
        elif st.session_state.pdf_file_name == pdf_file_uploader.name and \
             st.session_state.get("processing_complete_for_current_file", False):
            st.info(f"File '{pdf_file_uploader.name}' already processed.")

    elif "processed_data" in st.session_state and st.session_state.processed_data is not None:
        st.info(f"Previously processed: {st.session_state.processed_data['pdf_file_name']}. Upload a new file to reprocess.")
        if st.button("Clear Processed Data"):
            st.session_state.processed_data = None
            st.session_state.pdf_file_name = None
            st.session_state.processing_complete_for_current_file = False # Reset flag
            if "community_selectbox" in st.session_state:
                del st.session_state["community_selectbox"]
            if "trend_terms_input" in st.session_state:
                del st.session_state["trend_terms_input"]
            st.rerun()
    # Ensure the flag is reset if no file is uploaded initially or after clearing
    if not pdf_file_uploader and "processing_complete_for_current_file" not in st.session_state:
        st.session_state.processing_complete_for_current_file = False


@st.fragment
def display_visualization_section(app_config):
    if "processed_data" not in st.session_state or st.session_state.processed_data is None:
        st.info("Please upload and process a PDF first to see visualizations.")
        return

    data = st.session_state.processed_data
    chunks = data["chunks"]
    embeddings = data["embeddings"]
    communities = data["communities"]

    st.header("Analysis Results")
    st.caption(f"Based on: {data['pdf_file_name']} | Total processing time: {data['processing_time']:.2f} seconds")

    display_communities_info(communities, chunks)
    
    # --- Interactive Community Graph Visualization ---
    if communities and embeddings.shape[0] > 1:
        configured_n_neighbors = app_config.community_detector.n_neighbors
        max_possible_neighbors = embeddings.shape[0] - 1
        num_neighbors_for_vis = min(configured_n_neighbors, max_possible_neighbors)
        num_neighbors_for_vis = max(1, num_neighbors_for_vis)

        adj_matrix_for_vis = None
        if num_neighbors_for_vis > 0:
            adj_matrix_for_vis = create_adjacency_matrix(embeddings, num_neighbors_for_vis)

        if adj_matrix_for_vis is not None:
            st.subheader("Interactive Community Graph")
            with st.spinner("Generating interactive graph..."):
                graph_html_path = create_interactive_community_graph(
                    embeddings, adj_matrix_for_vis, communities, chunks, output_filename="community_graph.html"
                )

                if graph_html_path and os.path.exists(graph_html_path):
                    with open(graph_html_path, 'r', encoding='utf-8') as HtmlFile:
                        source_code = HtmlFile.read()
                    components.html(source_code, height=800, scrolling=True)
                elif graph_html_path:
                    st.error(f"Graph HTML file was expected at '{graph_html_path}' but not found.")
                else:
                    st.info("Could not generate community graph.")
        elif communities:
             st.info("Graph could not be generated (e.g., too few distinct data points).")
        elif not communities:
            st.info("No communities detected, so no community graph to display.")


    # --- Term Correlation Heatmap ---
    if communities:
        st.subheader("Term Correlation Heatmap")
        chunks_for_heatmap = {}
        for i, community_member_indices in enumerate(communities):
            if communities[i]:
                chunks_for_heatmap[i] = [chunks[chunk_idx] for chunk_idx in communities[i]]

        if not chunks_for_heatmap:
            st.info("No textual data in communities to generate heatmap.")
        else:
            with st.spinner("Generating correlation heatmap..."):
                heatmap_config = getattr(app_config, 'term_correlation_heatmap', {})
                top_n_val = heatmap_config.get('top_n_terms', 15) # Simplified
                heatmap_fig = plot_term_correlation_heatmap(chunks_for_heatmap, top_n_terms=top_n_val)
                if heatmap_fig:
                    st.pyplot(heatmap_fig)
                else:
                    st.info("Heatmap could not be generated.")

    # --- Temporal Trend Analysis ---
    st.subheader("Temporal Trend Analysis")
    if 'pdf_file_name' in data:
        try:
            report_date_str = data['pdf_file_name'].split('_')[-1].split('.')[0]
            report_date = pd.to_datetime(report_date_str)
        except Exception:
            report_date = pd.Timestamp('today')

        dated_chunks_data_for_trend = []
        if chunks:
            num_chunks = len(chunks)
            for i, chunk_text in enumerate(chunks):
                month_offset = int((i / num_chunks) * 12) if num_chunks > 0 else 0 # num_chunks > 1 to num_chunks > 0
                current_date = report_date - pd.DateOffset(months=month_offset)
                dated_chunks_data_for_trend.append({'text': chunk_text, 'date': current_date})

            if dated_chunks_data_for_trend:
                terms_input = st.text_input("Enter terms for trend (comma-separated):", "seo, keyword, ranking", key="trend_terms_input")
                if terms_input:
                    terms_to_plot = [term.strip() for term in terms_input.split(',') if term.strip()]
                    if terms_to_plot:
                        with st.spinner("Generating trend plot..."):
                            trend_fig = plot_term_trends(dated_chunks_data_for_trend, terms_to_plot)
                            if trend_fig:
                                st.pyplot(trend_fig)
                            else:
                                st.info("Trend plot could not be generated.")
                    else:
                        st.info("Please enter valid terms to track.")
            else:
                st.info("No data available for trend analysis.")
    else:
        st.info("PDF file name not available for temporal trend analysis.")


# Moved st.sidebar out of the fragment definition
# @st.fragment # This decorator applies to the function
def question_generation_fragment_content(app_config): # Renamed function to reflect it's content for the fragment
    if "processed_data" not in st.session_state or st.session_state.processed_data is None:
        st.info("Process a PDF to enable question generation.") # Will appear in sidebar
        return

    data = st.session_state.processed_data
    chunks = data["chunks"]
    communities = data["communities"]

    st.subheader("Generate Questions") # This will now correctly be inside the sidebar

    if not communities or len(communities) == 0:
        st.info("No communities detected to generate questions for.")
        return

    community_options = [f"Community {i+1} ({len(members)} items)" for i, members in enumerate(communities)]

    current_selection = st.session_state.get("community_selectbox")
    current_selection_idx = None
    if current_selection and current_selection in community_options:
        current_selection_idx = community_options.index(current_selection)

    selected_community_str = st.selectbox(
        "Select Community:",
        community_options,
        index=current_selection_idx if current_selection_idx is not None else 0,
        key="community_selectbox"
    )

    st.selectbox("LLM Generation Attempt Order:", ["OpenAI -> Ollama (Local)"], disabled=True, help="Fallback is automatic.")

    if selected_community_str:
        selected_idx = community_options.index(selected_community_str)
        selected_community_member_indices = communities[selected_idx]
        community_actual_texts = [chunks[i] for i in selected_community_member_indices]

        if st.button("Generate Questions for Selected Community", key=f"gen_q_comm_{selected_idx}"):
            if not community_actual_texts:
                st.warning("Selected community has no text to process.")
                return

            generated_questions = None
            used_llm = None
            openai_success = False

            with st.spinner("Attempting question generation with OpenAI..."):
                try:
                    cfg_openai = getattr(app_config, 'openai', {})
                    openai_gen = initialize_openai_question_generator(
                        model=getattr(cfg_openai, 'model', 'gpt-4o-mini'),
                        temperature=getattr(cfg_openai, 'temperature', 0.7),
                        max_tokens=getattr(cfg_openai, 'max_tokens', 200)
                    )
                    if openai_gen.client:
                        q_list = generate_questions_for_community(
                            openai_gen, community_actual_texts, selected_idx + 1, num_questions=3
                        )
                        if isinstance(q_list, list) and (not q_list or not q_list[0].startswith("Error:")):
                            generated_questions = q_list
                            used_llm = "OpenAI"
                            openai_success = True
                        else:
                            st.warning(f"OpenAI: {q_list[0] if q_list else 'Failed'}. Fallback...")
                    else:
                        st.warning("OpenAI client not init. Fallback...")
                except Exception as e:
                    st.warning(f"OpenAI error: {e}. Fallback...")

            if not openai_success:
                with st.spinner("Attempting question generation with Ollama..."):
                    try:
                        cfg_ollama = getattr(app_config, 'ollama', {})
                        ollama_gen = initialize_ollama_question_generator(
                            model=getattr(cfg_ollama, 'model', 'llama3.2:latest'),
                            host=getattr(cfg_ollama, 'host', None)
                        )
                        q_list = generate_questions_for_community(
                            ollama_gen, community_actual_texts, selected_idx + 1, num_questions=3
                        )
                        if isinstance(q_list, list) and (not q_list or not q_list[0].startswith("Error:")):
                            generated_questions = q_list
                            used_llm = "Ollama"
                            st.info("Questions generated using local Ollama model.")
                        else:
                            st.error(f"Ollama: {q_list[0] if q_list else 'Failed'}")
                            generated_questions = q_list
                    except Exception as e:
                        st.error(f"Ollama error: {e}. Both failed.")
                        generated_questions = [f"Error: Ollama exception: {e}"]

            # Display results in the sidebar under the button
            if generated_questions:
                st.subheader(f"Generated Qs (Community {selected_idx + 1}):")
                if used_llm: st.caption(f"(Using {used_llm})")
                if isinstance(generated_questions, list) and generated_questions and generated_questions[0].startswith("Error:"):
                    st.error("\n".join(generated_questions))
                else:
                    for q_idx, q_text in enumerate(generated_questions):
                        st.markdown(f"{q_idx + 1}. {q_text}")
            else:
                st.warning("No questions were generated.")


    # --- Batch Question Generation for Top 3 Communities ---
    st.markdown("---")

    if communities and len(communities) > 0:
        st.subheader("Batch Generate (Top 3)")
        if st.button("Generate for Top 3 (Async)", key="batch_gen_q"):
            batch_community_data = []
            for i, community_member_indices in enumerate(communities[:3]):
                if not community_member_indices: continue
                community_actual_texts = [chunks[i_chunk] for i_chunk in community_member_indices]
                if community_actual_texts:
                    batch_community_data.append({"id": f"Community {i+1}", "texts": community_actual_texts})

            if not batch_community_data:
                st.info("No communities with text found in the top 3.")
            else:
                with st.spinner("Generating questions for multiple communities asynchronously with OpenAI..."):
                    cfg_batch = getattr(app_config, 'openai', {})
                    openai_batch_gen = initialize_openai_question_generator(
                        model=getattr(cfg_batch, 'model', 'gpt-4o-mini'),
                        temperature=getattr(cfg_batch, 'temperature', 0.7),
                        max_tokens=getattr(cfg_batch, 'max_tokens', 200)
                    )

                    if not hasattr(openai_batch_gen, 'async_generate_questions_from_community_texts_batch') or \
                       not openai_batch_gen.async_client:
                        st.error("Async batch generation not supported.")
                    else:
                        batch_results = {}
                        try:
                            batch_results = asyncio.run(
                                openai_batch_gen.async_generate_questions_from_community_texts_batch(
                                    batch_community_data,
                                    num_questions_per_community=3
                                )
                            )
                        except Exception as e:
                            st.error(f"Async batch error: {e}")

                        if batch_results:
                            st.subheader("Batch Qs Results:")
                            for comm_id, q_list_res in batch_results.items():
                                st.markdown(f"**{comm_id}**")
                                if isinstance(q_list_res, list) and q_list_res and q_list_res[0].startswith("Error:"):
                                    st.error("\n".join(q_list_res))
                                elif not q_list_res:
                                    st.info(f"No Qs for {comm_id}.")
                                else:
                                    for q_idx, q_text in enumerate(q_list_res):
                                        st.markdown(f"- {q_text}")
                                st.markdown("---")
                        elif batch_community_data:
                            st.warning("Batch generation produced no results.")


@st.fragment
def display_search_section(app_config):
    if "processed_data" not in st.session_state or st.session_state.processed_data is None:
        st.info("Process a PDF to enable semantic search.")
        return

    data = st.session_state.processed_data
    chunks = data["chunks"]
    indexer = data["indexer"]
    embedder = data["embedder"]

    st.subheader("Semantic Search")
    if embedder and embedder.model is not None and indexer:
        query_text = st.text_input("Search for similar content:", key="search_query_input")
        if query_text:
            query_embedding = generate_embeddings(embedder, [query_text])

            if query_embedding is not None and query_embedding.shape[0] > 0:
                distances, indices = indexer.search(query_embedding, k=app_config.faiss_indexer.k_search)
                st.write(f"Search Results (Top {app_config.faiss_indexer.k_search}):")
                for i, idx in enumerate(indices[0]):
                    st.caption(f"Rank {i+1} (Dist: {distances[0][i]:.4f}): Chunk {idx} - {chunks[idx][:150]}...")
            else:
                st.warning("Could not generate embedding for the query.")
    else:
        st.warning("Search components not ready. Process a PDF.")


def main():
    # Load configurations
    app_config = load_config()

    st.set_page_config(page_title=app_config.streamlit_ui.title, layout="wide")
    st.title(app_config.streamlit_ui.title)

    # Initialize session state
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "pdf_file_name" not in st.session_state:
        st.session_state.pdf_file_name = None
    if "processing_complete_for_current_file" not in st.session_state:
        st.session_state.processing_complete_for_current_file = False


    # Display app sections
    display_processing_section(app_config) # This is a fragment, will run its content

    if st.session_state.processed_data is not None:
        display_visualization_section(app_config) # This is a fragment
        display_search_section(app_config)       # This is a fragment

        # For the question generation, which should be in the sidebar:
        # Call the content function within st.sidebar context, and that function is decorated with @st.fragment
        @st.fragment # Decorate the wrapper that calls the content function
        def display_question_generation_sidebar_wrapper(config): # Wrapper function
            question_generation_fragment_content(config) # Call the actual content function

        with st.sidebar: # Place the call to the wrapper fragment in the sidebar
            display_question_generation_sidebar_wrapper(app_config)
    else:
        st.info("Upload a PDF to begin analysis.")


if __name__ == '__main__':
    main()
