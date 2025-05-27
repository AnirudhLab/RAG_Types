import streamlit as st
import tempfile
from self_rag import load_documents, build_vectorstore, get_self_rag_chain

st.title("🧠 Self-RAG App (LangChain + Groq)")

uploaded_file = st.file_uploader("Upload a document (PDF/TXT)", type=["pdf", "txt"])
query = st.text_input("Ask a question")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    documents = load_documents(tmp_path)
    vectorstore = build_vectorstore(documents)
    chain = get_self_rag_chain(vectorstore)

    if query:
        result = chain.invoke({"query": query})
        st.markdown("### 🤖 Answer")
        st.write(result)
