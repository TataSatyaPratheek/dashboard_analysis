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
from dotenv import load_dotenv

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
def display_communities_info(communities, chunks, chunk_to_pdf_map): # Added chunk_to_pdf_map
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
                pdf_name = chunk_to_pdf_map[item_index] # Get source PDF
                st.caption(f"- Chunk {item_index} (from: {pdf_name}): {chunks[item_index][:150]}...")


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
    st.subheader("PDF Processing and Index Management")

    # Multiple file uploader
    pdf_files_uploader = st.file_uploader(
        "Upload SEO Report(s) (PDF)", 
        type="pdf",
        accept_multiple_files=True, 
        key="pdf_uploader_multiple"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Process New Files & Rebuild Global Index", key="process_rebuild_button"):
            if not pdf_files_uploader:
                st.warning("Please upload PDF files first.")
            else:
                new_files_processed_flag = False
                # Initialize embedder (can be cached)
                embedder = initialize_embedding_generator(app_config.embedding_generator.model_name)
                if embedder.model is None:
                    st.error("Failed to load embedding model. Cannot process PDFs.")
                    return # Stop processing if embedder fails

                for uploaded_file_obj in pdf_files_uploader:
                    file_bytes = uploaded_file_obj.getvalue()
                    file_name = uploaded_file_obj.name

                    if file_name not in st.session_state.processed_documents_data or \
                       st.session_state.processed_documents_data[file_name].get('embeddings') is None: # Process if new or failed previously
                        
                        st.info(f"Processing '{file_name}'...")
                        processor = initialize_pdf_processor(
                            app_config.pdf_processor.chunk_size,
                            app_config.pdf_processor.chunk_overlap
                        )
                        chunks = process_pdf(processor, file_bytes, file_name) # process_pdf uses filename for cache key

                        if not chunks:
                            st.error(f"No text extracted from '{file_name}'.")
                            st.session_state.processed_documents_data[file_name] = {'chunks': [], 'embeddings': None, 'name': file_name}
                            continue
                        
                        embeddings = generate_embeddings(embedder, chunks) # generate_embeddings is cached based on chunks
                        if embeddings is None or embeddings.shape[0] == 0:
                            st.error(f"No embeddings generated for '{file_name}'.")
                            st.session_state.processed_documents_data[file_name] = {'chunks': chunks, 'embeddings': None, 'name': file_name}
                            continue

                        st.session_state.processed_documents_data[file_name] = {
                            'chunks': chunks, 
                            'embeddings': embeddings,
                            'name': file_name
                        }
                        new_files_processed_flag = True
                        st.success(f"Successfully processed and generated embeddings for '{file_name}'.")
                    else:
                        st.info(f"'{file_name}' was already processed and loaded.")

                # Consolidate all data and (re)build global index if new files were processed or index doesn't exist
                if new_files_processed_flag or st.session_state.get("global_faiss_indexer") is None:
                    st.info("Consolidating data for global index...")
                    temp_all_chunks = []
                    temp_all_embeddings_list = []
                    temp_chunk_to_pdf_map = []

                    for fname, data_dict in st.session_state.processed_documents_data.items():
                        if data_dict.get('embeddings') is not None and data_dict['embeddings'].shape[0] > 0:
                            temp_all_chunks.extend(data_dict['chunks'])
                            temp_all_embeddings_list.append(data_dict['embeddings'])
                            temp_chunk_to_pdf_map.extend([fname] * len(data_dict['chunks']))
                    
                    if not temp_all_embeddings_list:
                        st.warning("No embeddings available from any processed PDF. Global index cannot be built.")
                    else:
                        st.session_state.global_all_chunks = temp_all_chunks
                        st.session_state.global_all_embeddings = np.vstack(temp_all_embeddings_list)
                        st.session_state.global_chunk_to_pdf_map = temp_chunk_to_pdf_map

                        st.info(f"Total embeddings for global index: {st.session_state.global_all_embeddings.shape[0]}")

                        # Build/Rebuild Global FAISS Index
                        dim = st.session_state.global_all_embeddings.shape[1]
                        
                        current_global_indexer = FAISSIndexer(
                            dim=dim,
                            m=app_config.faiss_indexer.m,
                            nbits=app_config.faiss_indexer.nbits
                        )
                        
                        current_global_indexer.build(st.session_state.global_all_embeddings)
                        st.session_state.global_faiss_indexer = current_global_indexer
                        st.success(f"Global FAISS index built/rebuilt successfully.")

                        # Detect Global Communities
                        if st.session_state.global_all_embeddings.shape[0] > 1:
                            n_neighbors_val = min(app_config.community_detector.n_neighbors, 
                                                  st.session_state.global_all_embeddings.shape[0] - 1)
                            n_neighbors_val = max(1, n_neighbors_val) 

                            detector = initialize_community_detector(
                                n_neighbors=n_neighbors_val,
                                use_weights=app_config.community_detector.use_weights
                            )
                            st.session_state.global_communities_detected = detect_communities(detector, st.session_state.global_all_embeddings)
                            if st.session_state.global_communities_detected:
                                st.success(f"Global communities detected: {len(st.session_state.global_communities_detected)} communities.")
                            else:
                                st.warning("Global community detection did not yield results (e.g., too few items or no distinct clusters).")
                        else:
                            st.session_state.global_communities_detected = None
                            st.info("Not enough data points for global community detection.")
                st.rerun() 

    with col2:
        if st.button("Clear All Processed Data & Index", key="clear_all_button"):
            keys_to_clear = [
                "processed_documents_data", "global_all_chunks", "global_all_embeddings",
                "global_faiss_indexer", "global_communities_detected", "global_chunk_to_pdf_map"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            if "pdf_uploader_multiple" in st.session_state:
                del st.session_state["pdf_uploader_multiple"]
            st.rerun()

    if st.session_state.processed_documents_data:
        st.subheader("Loaded Documents Summary:")
        for fname, data in st.session_state.processed_documents_data.items():
            num_c = len(data.get('chunks', []))
            emb_shape = data.get('embeddings').shape if data.get('embeddings') is not None else "N/A"
            st.markdown(f"- **{fname}**: {num_c} chunks, Embeddings: {emb_shape}")
    
    if st.session_state.get("global_faiss_indexer"):
        st.success(f"Global index is active with {st.session_state.global_faiss_indexer.index.ntotal} embeddings.")

@st.fragment
def display_visualization_section(app_config, global_chunks, global_embeddings, global_communities, global_chunk_map):
    if not global_chunks or global_embeddings.shape[0] == 0:
        st.info("Please upload and process PDF(s) first to see visualizations.")
        return

    st.header("Aggregated Analysis Results")
    total_processing_time = sum(d.get('processing_time', 0) for d in st.session_state.processed_documents_data.values())
    num_docs = len(st.session_state.processed_documents_data)
    st.caption(f"Based on {num_docs} document(s) | Total individual processing time: {total_processing_time:.2f} seconds")

    display_communities_info(global_communities, global_chunks, global_chunk_map)
    
    # --- Interactive Community Graph Visualization ---
    if global_communities and global_embeddings.shape[0] > 1:
        configured_n_neighbors = app_config.community_detector.n_neighbors
        max_possible_neighbors = global_embeddings.shape[0] - 1
        num_neighbors_for_vis = min(configured_n_neighbors, max_possible_neighbors)
        num_neighbors_for_vis = max(1, num_neighbors_for_vis)

        adj_matrix_for_vis = None
        if num_neighbors_for_vis > 0:
            adj_matrix_for_vis = create_adjacency_matrix(global_embeddings, num_neighbors_for_vis)

        if adj_matrix_for_vis is not None:
            st.subheader("Interactive Community Graph (Aggregated)")
            with st.spinner("Generating interactive graph..."):
                # TODO: Modify create_interactive_community_graph to accept chunk_to_pdf_map for richer tooltips
                graph_html_path = create_interactive_community_graph(
                    global_embeddings, adj_matrix_for_vis, global_communities, global_chunks,
                    global_chunk_map, # Pass the map
                    output_filename="community_graph.html"
                )

                if graph_html_path and os.path.exists(graph_html_path):
                    with open(graph_html_path, 'r', encoding='utf-8') as HtmlFile:
                        source_code = HtmlFile.read()
                    components.html(source_code, height=800, scrolling=True)
                elif graph_html_path:
                    st.error(f"Graph HTML file was expected at '{graph_html_path}' but not found.")
                else:
                    st.info("Could not generate community graph.")
        elif global_communities:
             st.info("Graph could not be generated (e.g., too few distinct data points).")
        elif not global_communities:
            st.info("No communities detected from aggregated data, so no community graph to display.")

    # --- Term Correlation Heatmap ---
    if global_communities:
        st.subheader("Term Correlation Heatmap (Aggregated)")
        chunks_for_heatmap = {}
        for i, community_member_indices in enumerate(global_communities):
            if global_communities[i]: # Check if community is not empty
                chunks_for_heatmap[i] = [global_chunks[chunk_idx] for chunk_idx in global_communities[i]]

        if not chunks_for_heatmap:
            st.info("No textual data in aggregated communities to generate heatmap.")
        else:
            with st.spinner("Generating correlation heatmap..."):
                heatmap_config = getattr(app_config, 'term_correlation_heatmap', {})
                top_n_val = heatmap_config.get('top_n_terms', 15)
                heatmap_fig = plot_term_correlation_heatmap(chunks_for_heatmap, top_n_terms=top_n_val)
                if heatmap_fig:
                    st.pyplot(heatmap_fig)
                else:
                    st.info("Heatmap could not be generated.")

    # --- Temporal Trend Analysis (Aggregated) ---
    st.subheader("Temporal Trend Analysis (Aggregated)")
    dated_chunks_data_for_trend_all_pdfs = []
    if st.session_state.processed_documents_data:
        for pdf_name, pdf_data in st.session_state.processed_documents_data.items():
            try:
                report_date_str = pdf_name.split('_')[-1].split('.')[0]
                report_date = pd.to_datetime(report_date_str)
            except Exception:
                report_date = pd.Timestamp('today') # Default date for this PDF

            pdf_chunks = pdf_data['chunks']
            num_pdf_chunks = len(pdf_chunks)
            for i, chunk_text in enumerate(pdf_chunks):
                month_offset = int((i / num_pdf_chunks) * 12) if num_pdf_chunks > 0 else 0
                current_date = report_date - pd.DateOffset(months=month_offset)
                dated_chunks_data_for_trend_all_pdfs.append({'text': chunk_text, 'date': current_date, 'source_pdf': pdf_name})
        
        if dated_chunks_data_for_trend_all_pdfs:
            terms_input = st.text_input("Enter terms for trend (comma-separated):", "seo, keyword, ranking", key="trend_terms_input_global")
            if terms_input:
                terms_to_plot = [term.strip() for term in terms_input.split(',') if term.strip()]
                if terms_to_plot:
                    with st.spinner("Generating aggregated trend plot..."):
                        trend_fig = plot_term_trends(dated_chunks_data_for_trend_all_pdfs, terms_to_plot)
                        if trend_fig:
                            st.pyplot(trend_fig)
                        else:
                            st.info("Aggregated trend plot could not be generated.")
                else:
                    st.info("Please enter valid terms to track.")
        else:
            st.info("No data available for aggregated trend analysis.")
    else:
        st.info("No PDF documents processed for temporal trend analysis.")


@st.fragment
def question_generation_fragment_content(app_config):
    # This fragment will now use global session state data
    if not st.session_state.get("global_all_chunks") or not st.session_state.get("global_communities_detected"):
        st.info("Process PDF(s) to enable question generation from aggregated data.")
        return

    global_chunks = st.session_state.global_all_chunks
    global_communities = st.session_state.global_communities_detected

    st.subheader("Generate Questions (Aggregated)")

    if not global_communities or len(global_communities) == 0:
        st.info("No communities detected from aggregated data to generate questions for.")
        return

    community_options = [f"Community {i+1} ({len(members)} items)" for i, members in enumerate(global_communities)]

    current_selection = st.session_state.get("community_selectbox_global")
    current_selection_idx = None
    if current_selection and current_selection in community_options:
        current_selection_idx = community_options.index(current_selection)

    selected_community_str = st.selectbox(
        "Select Aggregated Community:",
        community_options,
        index=current_selection_idx if current_selection_idx is not None else 0,
        key="community_selectbox_global"
    )

    st.selectbox("LLM Generation Attempt Order:", ["OpenAI -> Ollama (Local)"], disabled=True, help="Fallback is automatic.")

    if selected_community_str:
        selected_idx = community_options.index(selected_community_str)
        selected_community_member_indices = global_communities[selected_idx]
        # Ensure indices are valid for global_chunks
        community_actual_texts = [global_chunks[i] for i in selected_community_member_indices if i < len(global_chunks)]


        if st.button("Generate Questions for Selected Aggregated Community", key=f"gen_q_comm_global_{selected_idx}"):
            if not community_actual_texts:
                st.warning("Selected aggregated community has no text to process.")
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
                            openai_gen, community_actual_texts, f"Aggregated Community {selected_idx + 1}", num_questions=3
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
                            ollama_gen, community_actual_texts, f"Aggregated Community {selected_idx + 1}", num_questions=3
                        )
                        if isinstance(q_list, list) and (not q_list or not q_list[0].startswith("Error:")):
                            generated_questions = q_list
                            used_llm = "Ollama"
                            st.info("Questions generated using local Ollama model.")
                        else:
                            st.error(f"Ollama: {q_list[0] if q_list else 'Failed'}")
                            generated_questions = q_list # Store error message
                    except Exception as e:
                        st.error(f"Ollama error: {e}. Both failed.")
                        generated_questions = [f"Error: Ollama exception: {e}"]

            if generated_questions:
                st.subheader(f"Generated Qs (Agg. Comm. {selected_idx + 1}):")
                if used_llm: st.caption(f"(Using {used_llm})")
                if isinstance(generated_questions, list) and generated_questions and generated_questions[0].startswith("Error:"):
                    st.error("\n".join(generated_questions))
                else:
                    for q_idx, q_text in enumerate(generated_questions):
                        st.markdown(f"{q_idx + 1}. {q_text}")
            else:
                st.warning("No questions were generated for the aggregated community.")

    st.markdown("---")
    if global_communities and len(global_communities) > 0:
        st.subheader("Batch Generate (Top 3 Aggregated)")
        if st.button("Generate for Top 3 Agg. (Async)", key="batch_gen_q_global"):
            batch_community_data = []
            num_to_process = min(len(global_communities), 3)
            for i in range(num_to_process):
                community_member_indices = global_communities[i]
                if not community_member_indices: continue
                community_actual_texts = [global_chunks[j] for j in community_member_indices if j < len(global_chunks)]
                if community_actual_texts:
                    batch_community_data.append({"id": f"Aggregated Community {i+1}", "texts": community_actual_texts})

            if not batch_community_data:
                st.info("No aggregated communities with text found in the top 3.")
            else:
                with st.spinner("Generating questions for multiple aggregated communities asynchronously with OpenAI..."):
                    cfg_batch = getattr(app_config, 'openai', {})
                    openai_batch_gen = initialize_openai_question_generator(
                        model=getattr(cfg_batch, 'model', 'gpt-4o-mini'),
                        temperature=getattr(cfg_batch, 'temperature', 0.7),
                        max_tokens=getattr(cfg_batch, 'max_tokens', 200)
                    )
                    if not hasattr(openai_batch_gen, 'async_generate_questions_from_community_texts_batch') or \
                       not openai_batch_gen.async_client:
                        st.error("Async batch generation not supported by the current OpenAI generator setup.")
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
                            st.error(f"Async batch error for aggregated communities: {e}")

                        if batch_results:
                            st.subheader("Batch Qs Results (Aggregated):")
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
                        elif batch_community_data: # If we tried to process but got no results
                            st.warning("Batch generation for aggregated communities produced no results.")


@st.fragment
def display_search_section(app_config, global_chunks, global_indexer, global_chunk_map):
    if not global_chunks or not global_indexer or not global_indexer.is_trained or global_indexer.index.ntotal == 0 :
        st.info("Process PDF(s) to enable semantic search on aggregated data.")
        return

    st.subheader("Semantic Search (Aggregated)")
    embedder = initialize_embedding_generator(app_config.embedding_generator.model_name)

    if embedder and embedder.model is not None:
        query_text = st.text_input("Search across all documents:", key="search_query_input_global")
        if query_text:
            query_embedding = generate_embeddings(embedder, [query_text])

            if query_embedding is not None and query_embedding.shape[0] > 0:
                distances, indices = global_indexer.search(query_embedding, k=app_config.faiss_indexer.k_search)
                if indices.size > 0:
                    st.write(f"Search Results (Top {app_config.faiss_indexer.k_search} from aggregated data):")
                    for i, idx in enumerate(indices[0]):
                        pdf_name_for_result = global_chunk_map[idx] if global_chunk_map and idx < len(global_chunk_map) else "Unknown Source"
                        st.caption(f"Rank {i+1} (Dist: {distances[0][i]:.4f}): Chunk {idx} (from: {pdf_name_for_result}) - {global_chunks[idx][:150]}...")
                else:
                    st.info("No results found for your query in the aggregated data.")
            else:
                st.warning("Could not generate embedding for the query.")
    else:
        st.warning("Embedding model not ready. Cannot perform search.")


def main():
    app_config = load_config()
    dotenv_path = "/Users/vi/Documents/work/dashboard_analysis/serious/.env" # Ensure this path is correct or use relative path
    load_dotenv(dotenv_path=dotenv_path)

    st.set_page_config(page_title=app_config.streamlit_ui.title, layout="wide")
    st.title(app_config.streamlit_ui.title)

    # Initialize global session state variables for multi-PDF processing
    if "processed_documents_data" not in st.session_state:
        st.session_state.processed_documents_data = {} # Stores {'filename': {'chunks': [], 'embeddings': ndarray, 'name': str, 'processing_time': float}}
    if "global_all_chunks" not in st.session_state:
        st.session_state.global_all_chunks = []
    if "global_all_embeddings" not in st.session_state:
        st.session_state.global_all_embeddings = np.array([])
    if "global_faiss_indexer" not in st.session_state:
        st.session_state.global_faiss_indexer = None
    if "global_communities_detected" not in st.session_state:
        st.session_state.global_communities_detected = None
    if "global_chunk_to_pdf_map" not in st.session_state:
        st.session_state.global_chunk_to_pdf_map = []
    if "trigger_global_rebuild" not in st.session_state: # Flag to signal need for global data aggregation
        st.session_state.trigger_global_rebuild = False
    
    display_processing_section(app_config)

    # Check if global data is ready for display and further analysis
    if st.session_state.get("global_faiss_indexer") is not None and \
       st.session_state.global_all_chunks and \
       st.session_state.global_all_embeddings.size > 0:

        display_visualization_section(app_config,
                                      st.session_state.global_all_chunks,
                                      st.session_state.global_all_embeddings,
                                      st.session_state.global_communities_detected,
                                      st.session_state.global_chunk_to_pdf_map)
        
        display_search_section(app_config,
                               st.session_state.global_all_chunks,
                               # global_all_embeddings is not directly needed by search if indexer is passed
                               # embedder is initialized within display_search_section
                               st.session_state.global_faiss_indexer,
                               st.session_state.global_chunk_to_pdf_map)

        @st.fragment
        def display_question_generation_sidebar_wrapper(config):
            question_generation_fragment_content(config)

        with st.sidebar:
            display_question_generation_sidebar_wrapper(app_config)
    elif st.session_state.processed_documents_data: # If some docs are processed but global state not ready (e.g. error during rebuild)
        st.warning("Aggregated data is being prepared or encountered an issue. Please wait or check processing status.")
    else:
        st.info("Upload and process PDF(s) to begin analysis.")

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
            # Iterate up to the first 3 communities
            num_to_process = min(len(communities), 3)
            for i in range(num_to_process):
                community_member_indices = communities[i] # igraph.VertexClustering supports integer indexing
                if not community_member_indices: continue # A community (list of indices) could be empty
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

    # Load environment variables from .env file
    # The .env file should be at /Users/vi/Documents/work/dashboard_analysis/serious/.env
    dotenv_path = "/Users/vi/Documents/work/dashboard_analysis/serious/.env"
    load_dotenv(dotenv_path=dotenv_path)

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
