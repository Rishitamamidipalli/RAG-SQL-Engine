import streamlit as st
from rag_system import RAGSQLEngine


def init_rag() -> RAGSQLEngine:
    """Initialize and return the singleton RAG system."""
    rag = RAGSQLEngine()
    return rag


def main():
    st.set_page_config(page_title="SQL RAG Assistant", page_icon="🤖", layout="wide")
    st.title("SQL RAG Assistant")
    st.write("Ask a question in natural language and get a SQL query generated from your knowledge base context.")

    rag = init_rag()

    with st.sidebar:
        st.header("Status")
        if rag.is_initialized():
            st.success("RAG initialized")
            # Show collection info
            info = rag.get_collection_info()
            if "error" not in info:
                st.info(f"Vectors in DB: {info.get('vectors_count', 0)}")
            else:
                st.warning("Could not get collection info")
        else:
            st.error("RAG not initialized")

        st.markdown("---")
        st.caption("Tip: Ensure HCL_API_KEY is set in .env for SQL generation.")

    query = st.text_area("Your question", height=120, placeholder="e.g., List the top 10 customers with the highest transaction amount.")

    col1, col2 = st.columns([1, 1])
    with col1:
        generate = st.button("Generate SQL", type="primary")
    with col2:
        show_context = st.checkbox("Show retrieved context", value=True)

    if generate:
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving context and generating SQL..."):
            # Increased to top_k=15 for comprehensive context retrieval
            context_results = rag.search_similar_documents(query, top_k=5)
            sql = rag.generate_rag_response(query, top_k=5)

        st.subheader("Generated SQL")
        if sql:
            st.code(sql, language="sql")
        else:
            st.error("Failed to generate SQL. Check logs and API key configuration.")

        if show_context:
            st.subheader("Retrieved Context Snippets")
            if context_results:
                for idx, res in enumerate(context_results, start=1):
                    with st.expander(f"Snippet {idx} (score: {res['score']:.3f})"):
                        st.write(res["content"])
                        st.caption(f"Source: {res['source']} | Page: {res.get('page', 0)} | Chunk: {res.get('chunk_index', 0)}")
            else:
                st.info("No relevant context found.")


if __name__ == "__main__":
    main()

