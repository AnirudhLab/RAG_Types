import streamlit as st
import tempfile
from rag_chain import load_documents, create_vector_store, get_qa_chain

st.title("📄 Simple RAG App (Groq + LangChain)")

uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
query = st.text_input("Ask a question about the document")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document..."):
        documents = load_documents(tmp_path)
        vector_store = create_vector_store(documents)
        qa_chain = get_qa_chain(vector_store)

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain(query)
            st.markdown("### 💬 Answer")
            st.write(result["result"])

            st.markdown("### 📄 Source Snippets")
            for doc in result["source_documents"]:
                st.write(doc.page_content[:300] + "...")
