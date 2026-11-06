import os
import streamlit as st
import pandas as pd
from config import DATA_DIR, GROQ_API_KEY, DEFAULT_LLM_MODEL, CHROMA_DIR
from ingestion.loader import load_single_document, get_document_metadata, get_file_hash
from ingestion.chunker import split_documents
from database.chroma_manager import ChromaManager
from rag.retriever import retrieve_relevant_chunks
from rag.generator import AnswerGenerator
from rag.query_processor import rewrite_query
from history.chat_manager import ChatManager
from ingestion.document_manager import DocumentManager
from evaluation.evaluator import RAGEvaluator

# App Title & Layout Configuration
st.set_page_config(
    page_title="Intelligent Knowledge Assistant (RAG)",
    page_icon="ðŸŽ“",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* CSS Styling updates */
    .reportview-container {
        background: #0F172A;
        color: #F8FAFC;
    }
    .stSidebar {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    h1, h2, h3, h4 {
        color: #38BDF8 !important;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stButton>button {
        background-color: #0284C7;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #38BDF8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
    }
    .chat-bubble {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    .chat-bubble.user {
        background-color: #1E293B;
    }
    .chat-bubble.assistant {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
    }
    .source-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
    }
    .suggestion-chip {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 16px;
        background-color: #1E293B;
        border: 1px solid #334155;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    .suggestion-chip:hover {
        background-color: #38BDF8;
        color: #0F172A;
        border-color: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize persistent classes
@st.cache_resource
def get_managers():
    cm = ChromaManager()
    chm = ChatManager(CHROMA_DIR)
    dm = DocumentManager(CHROMA_DIR)
    return cm, chm, dm

chroma_manager, chat_manager, doc_manager = get_managers()

# Auto register existing project documents on first run if they exist but are not tracked
existing_files = chroma_manager.get_ingested_files()
registry_docs = doc_manager.get_all_documents()
for fhash, finfo in existing_files.items():
    fname = finfo["filename"]
    if fname not in registry_docs:
        local_path = os.path.join(DATA_DIR, fname)
        size_kb = 0.0
        ext = os.path.splitext(fname)[1].lower()
        if os.path.exists(local_path):
            size_kb = round(os.path.getsize(local_path) / 1024, 2)
        doc_manager.register_document(fname, fhash, size_kb, finfo["chunk_count"], ext)

# Sync deletes: if database was reset or registry contains files not in Chroma
for fname in list(registry_docs.keys()):
    found_in_chroma = False
    for fhash, finfo in existing_files.items():
        if finfo["filename"] == fname:
            found_in_chroma = True
            break
    if not found_in_chroma:
        doc_manager.remove_document(fname)

# SIDEBAR PANELS
with st.sidebar:
    st.markdown("### ðŸŽ“ Intelligent Knowledge Assistant")
    st.markdown("CSE Capstone Project Dashboard")

    st.markdown("---")

    # Navigation controls
    st.subheader("ðŸ“ Scope Selector")
    all_files = ["All Documents"] + list(doc_manager.get_all_documents().keys())
    selected_doc_scope = st.selectbox(
        "Grounding Scope Context",
        options=all_files,
        help="Ground assistant answers ONLY within the selected document."
    )

    st.markdown("---")

    # Chat history sessions section
    st.subheader("ðŸ’¬ Chat Session History")

    if st.button("âž• Start New Chat", use_container_width=True):
        new_sid = chat_manager.create_session("New Chat")
        st.session_state.active_session_id = new_sid
        st.rerun()

    # Load sessions
    sessions = chat_manager.get_all_sessions()
    if not sessions:
        # Create a default session if none exists
        default_sid = chat_manager.create_session("New Chat")
        st.session_state.active_session_id = default_sid
        sessions = chat_manager.get_all_sessions()

    if "active_session_id" not in st.session_state or st.session_state.active_session_id not in sessions:
        st.session_state.active_session_id = list(sessions.keys())[-1]

    active_sid = st.session_state.active_session_id

    # List sessions dynamically
    for sid, info in list(sessions.items()):
        col_title, col_action = st.columns([4, 1])
        btn_label = f"ðŸ’¬ {info['title']}"
        if sid == active_sid:
            btn_label = f"ðŸ‘‰ {info['title']}"

        if col_title.button(btn_label, key=f"session_btn_{sid}", use_container_width=True):
            st.session_state.active_session_id = sid
            st.rerun()

        if col_action.button("ðŸ—‘ï¸", key=f"session_del_{sid}"):
            chat_manager.delete_session(sid)
            if sid == active_sid:
                st.session_state.active_session_id = list(chat_manager.get_all_sessions().keys())[-1] if chat_manager.get_all_sessions() else None
            st.rerun()

    # Rename active session option
    if active_sid and active_sid in sessions:
        with st.expander("âœï¸ Rename Current Session"):
            new_title_input = st.text_input("New Session Title", value=sessions[active_sid]["title"])
            if st.button("Save Title"):
                chat_manager.rename_session(active_sid, new_title_input)
                st.rerun()

    st.markdown("---")

    # Quick clear history button
    if st.button("âš ï¸ Clear All Chat History", use_container_width=True):
        chat_manager.clear_all_sessions()
        st.success("Chat history cleared!")
        st.rerun()

# MAIN WEB LAYOUT Tabs
tab_chat, tab_docs, tab_eval, tab_admin = st.tabs(["ðŸ’¬ Chat Assistant", "ðŸ“ Document Hub", "ðŸ§ª RAG Evaluation", "âš™ï¸ Admin Panel"])

# Retrieve developer parameters (admin values are retrieved from session state or config default)
if "developer_chunk_size" not in st.session_state:
    st.session_state.developer_chunk_size = 500
if "developer_chunk_overlap" not in st.session_state:
    st.session_state.developer_chunk_overlap = 100
if "developer_top_k" not in st.session_state:
    st.session_state.developer_top_k = 4
if "developer_threshold" not in st.session_state:
    st.session_state.developer_threshold = 0.35
if "developer_llm" not in st.session_state:
    st.session_state.developer_llm = "openai/gpt-oss-20b"

# Setup default test variables
active_llm = st.session_state.developer_llm
dev_top_k = st.session_state.developer_top_k
dev_threshold = st.session_state.developer_threshold

# TAB 1: Chat Assistant
with tab_chat:
    st.title("ðŸ¤– Intelligent Knowledge Assistant")
    st.caption("CSE Capstone Project â€” Grounded Document Factual Q&A System.")

    # Suggested Prompts (dynamic based on selected document scope)
    st.write("### ðŸ’¡ Suggested Prompts")
    suggested_cols = st.columns(3)

    # Determine suggestion prompts based on the selected document scope
    suggestions = [
        "What is the definition of RAG?",
        "What are the three steps in standard RAG?",
        "What is ChromaDB?"
    ]
    if "AI_SmartLoan_Project_Documentation_Final.docx" in selected_doc_scope:
        suggestions = [
            "What is the problem statement?",
            "What are the objectives?",
            "Explain the document validation workflow."
        ]
    elif "project_docs.txt" in selected_doc_scope:
        suggestions = [
            "What is sentence-transformers/all-MiniLM-L6-v2?",
            "What role does Groq play in this project?",
            "What are the potential future enhancements?"
        ]

    for idx, prompt_text in enumerate(suggestions):
        if suggested_cols[idx].button(prompt_text, key=f"suggest_{idx}", use_container_width=True):
            # Process prompt
            if active_sid:
                chat_manager.add_message(active_sid, "user", prompt_text)

                with st.spinner("Retrieving source documents & generating answer..."):
                    history_messages = sessions[active_sid]["messages"] if active_sid in sessions else []

                    # Rewrite query for conversational continuity
                    standalone_query = rewrite_query(prompt_text, history_messages, GROQ_API_KEY, active_llm)

                    # Retrieve relevant segments
                    chunks = retrieve_relevant_chunks(
                        standalone_query,
                        chroma_manager,
                        top_k=dev_top_k,
                        similarity_threshold=dev_threshold,
                        document_scope=selected_doc_scope
                    )

                    # Generate response
                    if not GROQ_API_KEY:
                        ans_text = "Configuration Error: Groq API Key is missing. Please set the GROQ_API_KEY."
                        sources_res = []
                    else:
                        try:
                            generator = AnswerGenerator(model_name=active_llm, api_key=GROQ_API_KEY)
                            res = generator.generate_answer(standalone_query, chunks)
                            ans_text = res["answer"]
                            sources_res = res["sources"]
                        except Exception as e:
                            ans_text = f"Error during generation: {str(e)}"
                            sources_res = []

                    chat_manager.add_message(active_sid, "assistant", ans_text, sources_res)
                    st.rerun()

    st.markdown("---")

    # Render Conversation Messages
    if active_sid and active_sid in sessions:
        for m_idx, msg in enumerate(sessions[active_sid]["messages"]):
            role = msg["role"]
            with st.chat_message(role):
                st.write(msg["content"])

                # Render clean expandable source reference cards
                if role == "assistant" and msg.get("sources"):
                    with st.expander("ðŸ“ View Grounding Sources"):
                        for s_idx, src in enumerate(msg["sources"]):
                            st.markdown(f"**Source {s_idx+1}:** {src['source']} | Page: `{src['page']}` | Relevance Match: `{round(src['score']*100, 1)}%`")
                            st.info(src["text"])

                # Edit User question feature
                if role == "user":
                    with st.expander("âœï¸ Edit this Question"):
                        edit_q_input = st.text_input("Modify Question text", value=msg["content"], key=f"edit_q_val_{m_idx}")
                        if st.button("Regenerate Answer", key=f"edit_q_btn_{m_idx}"):
                            # Re-write this message and clear everything following it
                            sessions[active_sid]["messages"] = sessions[active_sid]["messages"][:m_idx]
                            chat_manager.add_message(active_sid, "user", edit_q_input)

                            # Re-run pipeline
                            with st.spinner("Regenerating response..."):
                                standalone_query = rewrite_query(edit_q_input, sessions[active_sid]["messages"][:-1], GROQ_API_KEY, active_llm)
                                chunks = retrieve_relevant_chunks(
                                    standalone_query,
                                    chroma_manager,
                                    top_k=dev_top_k,
                                    similarity_threshold=dev_threshold,
                                    document_scope=selected_doc_scope
                                )
                                generator = AnswerGenerator(model_name=active_llm, api_key=GROQ_API_KEY)
                                res = generator.generate_answer(standalone_query, chunks)
                                chat_manager.add_message(active_sid, "assistant", res["answer"], res["sources"])
                            st.rerun()

    # Chat input message text bar
    if user_input := st.chat_input("Enter your query..."):
        if active_sid:
            chat_manager.add_message(active_sid, "user", user_input)

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing knowledge context..."):
                    history_messages = sessions[active_sid]["messages"][:-1] if active_sid in sessions else []

                    # Context query rewriting
                    standalone_query = rewrite_query(user_input, history_messages, GROQ_API_KEY, active_llm)

                    # Search
                    chunks = retrieve_relevant_chunks(
                        standalone_query,
                        chroma_manager,
                        top_k=dev_top_k,
                        similarity_threshold=dev_threshold,
                        document_scope=selected_doc_scope
                    )

                    # Generate
                    if not GROQ_API_KEY:
                        ans_text = "Configuration Error: Groq API Key is missing. Please set the GROQ_API_KEY."
                        sources_res = []
                        st.write(ans_text)
                    else:
                        try:
                            generator = AnswerGenerator(model_name=active_llm, api_key=GROQ_API_KEY)
                            res = generator.generate_answer(standalone_query, chunks)
                            ans_text = res["answer"]
                            sources_res = res["sources"]

                            st.write(ans_text)
                            if sources_res:
                                with st.expander("ðŸ“ View Grounding Sources"):
                                    for s_idx, src in enumerate(sources_res):
                                        st.markdown(f"**Source {s_idx+1}:** {src['source']} | Page: `{src['page']}` | Relevance Match: `{round(src['score']*100, 1)}%`")
                                        st.info(src["text"])
                        except Exception as e:
                            ans_text = f"Error during generation: {str(e)}"
                            sources_res = []
                            st.write(ans_text)

                    chat_manager.add_message(active_sid, "assistant", ans_text, sources_res)
            st.rerun()

# TAB 2: Document Hub
with tab_docs:
    st.title("ðŸ“ Document Ingestion Hub")
    st.markdown("Upload files (PDF, TXT, DOCX) to expand the knowledge registry database.")

    # Ingestion configuration controls
    chunk_size = st.session_state.developer_chunk_size
    chunk_overlap = st.session_state.developer_chunk_overlap

    # Local Uploader
    uploaded_files = st.file_uploader(
        "Select files to ingest",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("âš¡ Index Uploaded Documents", key="ingest_btn_main", use_container_width=True):
            for uf in uploaded_files:
                local_path = os.path.join(DATA_DIR, uf.name)

                # Write file locally
                with open(local_path, "wb") as f:
                    f.write(uf.getbuffer())

                file_hash = get_file_hash(local_path)
                file_size_kb = round(os.path.getsize(local_path) / 1024, 2)
                file_type = os.path.splitext(uf.name)[1].lower()

                # Check duplicates
                if chroma_manager.is_file_ingested(file_hash):
                    st.warning(f"File '{uf.name}' already exists in index database. Ingestion skipped.")
                    continue

                with st.spinner(f"Processing '{uf.name}'..."):
                    try:
                        docs = load_single_document(local_path)
                        chunks = split_documents(docs, chunk_size, chunk_overlap)

                        # Add to vectorDB index
                        if chroma_manager.add_documents(chunks, uf.name, file_hash):
                            doc_manager.register_document(uf.name, file_hash, file_size_kb, len(chunks), file_type)
                            st.success(f"Indexed document '{uf.name}' successfully ({len(chunks)} chunks).")
                    except Exception as e:
                        st.error(f"Failed to index '{uf.name}': {str(e)}")
            st.rerun()

    st.markdown("---")

    # Render registered documents
    st.subheader("Indexed Documents")
    registry = doc_manager.get_all_documents()
    if not registry:
        st.info("No documents indexed. Ingest a document above.")
    else:
        for fname, info in list(registry.items()):
            col_icon, col_details, col_actions = st.columns([1, 8, 3])

            icon = "ðŸ“„"
            if info["file_type"] == ".pdf":
                icon = "ðŸ“•"
            elif info["file_type"] == ".docx":
                icon = "ðŸ“˜"

            col_icon.markdown(f"## {icon}")
            col_details.markdown(f"**{fname}**  \nSize: `{info['file_size_kb']} KB` | Chunks: `{info['chunk_count']}` | Status: `{info['status']}`")

            # Action controls (Preview text / Delete)
            col_preview, col_delete = col_actions.columns(2)

            # Preview Option
            if col_preview.button("ðŸ”Ž Preview", key=f"prev_{fname}"):
                local_path = os.path.join(DATA_DIR, fname)
                if os.path.exists(local_path):
                    try:
                        docs = load_single_document(local_path)
                        st.session_state.preview_doc_name = fname
                        st.session_state.preview_doc_text = "\n\n".join([d.page_content[:600] + "..." for d in docs[:3]])
                    except Exception as e:
                        st.error(f"Error previewing: {str(e)}")
                else:
                    st.error("Local file missing.")

            # Delete Option
            if col_delete.button("ðŸ—‘ï¸ Delete", key=f"del_confirm_{fname}"):
                st.session_state.delete_target = fname

        # Confirm Deletion block
        if "delete_target" in st.session_state:
            target = st.session_state.delete_target
            st.warning(f"Are you sure you want to permanently delete '{target}' from the database and registry?")
            col_yes, col_no = st.columns(2)
            if col_yes.button("Yes, Delete", key="btn_confirm_del_yes"):
                with st.spinner(f"Deleting '{target}'..."):
                    # Delete from Chroma vector space
                    chroma_manager.delete_document(target)
                    # Delete from local metadata file register
                    doc_manager.remove_document(target)
                    # Delete local raw file if exists
                    local_path = os.path.join(DATA_DIR, target)
                    if os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except Exception:
                            pass
                del st.session_state.delete_target
                st.success(f"Document '{target}' deleted successfully.")
                st.rerun()
            if col_no.button("Cancel", key="btn_confirm_del_no"):
                del st.session_state.delete_target
                st.rerun()

        # Render preview card if active
        if "preview_doc_name" in st.session_state:
            st.markdown("---")
            st.markdown(f"#### ðŸ” Previewing: {st.session_state.preview_doc_name}")
            st.code(st.session_state.preview_doc_text, language="markdown")
            if st.button("Close Preview"):
                del st.session_state.preview_doc_name
                del st.session_state.preview_doc_text
                st.rerun()

# TAB 3: RAG Evaluation
with tab_eval:
    st.title("ðŸ§ª Automated RAG Performance Evaluation")
    st.markdown("Run the evaluation suite comprising grounded queries, multi-hop lookups, and out-of-domain prompts.")

    if st.button("ðŸš€ Run System Evaluation", key="eval_suite_btn_tab", use_container_width=True):
        if not GROQ_API_KEY:
            st.error("Cannot run evaluation without a valid Groq API Key.")
        else:
            with st.spinner("Evaluating test queries..."):
                try:
                    evaluator = RAGEvaluator()
                    summary = evaluator.run_evaluation(chroma_manager, dev_top_k, dev_threshold)

                    if "error" in summary:
                        st.error(summary["error"])
                    else:
                        st.success("Evaluation completed successfully!")

                        # Summary Metrics
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Total Test Cases", summary["total_questions"])
                        m_col2.metric("Correct Refusals", summary["correct_refusals"])
                        m_col3.metric("Correctly Answered", summary["supported_answered"])

                        m2_col1, m2_col2, m2_col3 = st.columns(3)
                        m2_col1.metric("Hallucinations (Failed Refusal)", summary["unsupported_failed_refusal_count"])
                        m2_col2.metric("Misses (Failed Answers)", summary["supported_failed_answered_count"])
                        m2_col3.metric("Avg Response Latency", f"{summary['average_latency_seconds']} s")

                        # Detailed results table
                        st.subheader("Detailed Evaluation Log")
                        for res in summary["results"]:
                            status_color = "ðŸŸ¢" if "Correct" in res["status"] or "Successfully" in res["status"] else "ðŸ”´"

                            with st.expander(f"{status_color} Q: {res['question']} ({res['expected_category']})"):
                                st.markdown(f"**Status:** `{res['status']}`")
                                st.markdown(f"**Latency:** `{res['latency_seconds']} seconds`")
                                st.markdown(f"**Retrieved Sources:** `{res['sources_retrieved']}`")
                                st.markdown("**Answer Output:**")
                                st.code(res["actual_answer"], language="markdown")
                except Exception as e:
                    st.error(f"Error running evaluation: {str(e)}")

# TAB 4: Admin Panel (Technical Settings)
with tab_admin:
    st.title("âš™ï¸ Developer & Admin Settings Console")
    st.markdown("Exposes internal technical configurations and database logs for developer debugging.")

    # Setup configuration toggle validation
    is_admin = st.checkbox("Unlock Developer Settings", value=False)

    if is_admin:
        st.subheader("âš™ï¸ Parameter Configurations")

        col_c1, col_c2 = st.columns(2)
        dev_chunk = col_c1.slider("Index Chunk Size (Characters)", min_value=200, max_value=1500, value=chunk_size, step=50)
        dev_overlap = col_c2.slider("Index Chunk Overlap", min_value=0, max_value=400, value=chunk_overlap, step=25)

        col_r1, col_r2 = st.columns(2)
        dev_k = col_r1.slider("Retrieval Top K Documents", min_value=1, max_value=10, value=dev_top_k, step=1)
        dev_thresh = col_r2.slider("Retrieval Similarity Threshold", min_value=0.0, max_value=1.0, value=dev_threshold, step=0.05)

        dev_llm_sel = st.selectbox(
            "API Inference Model Endpoint",
            options=["openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
            index=["openai/gpt-oss-20b", "qwen/qwen3.6-27b"].index(active_llm)
        )

        if st.button("ðŸ’¾ Save Configurations"):
            st.session_state.developer_chunk_size = dev_chunk
            st.session_state.developer_chunk_overlap = dev_overlap
            st.session_state.developer_top_k = dev_k
            st.session_state.developer_threshold = dev_thresh
            st.session_state.developer_llm = dev_llm_sel
            st.success("Configurations updated successfully!")
            st.rerun()

        st.markdown("---")

        st.subheader("ðŸ“Š Vector DB Properties")
        st.write(f"- Embeddings model: `sentence-transformers/all-MiniLM-L6-v2`")
        st.write(f"- DB persistent path: `{CHROMA_DIR}`")
        st.write(f"- API Status: `Configured`" if GROQ_API_KEY else "API Status: `Missing API Key`")

        if st.button("ðŸ—‘ï¸ Reset Database Index & Registry", key="admin_reset_db_btn"):
            if chroma_manager.reset_database():
                doc_manager.clear_all_sessions() # Clear tracked docs too
                st.success("Database fully cleared.")
                st.rerun()
            else:
                st.error("Failed to reset database.")
    else:
        st.info("Unlock developer settings to view configuration inputs.")
