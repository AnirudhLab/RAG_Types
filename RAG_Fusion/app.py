import streamlit as st
import tempfile

import sys
import numpy

import streamlit as st
st.write("Python executable:", sys.executable)
st.write("NumPy version:", numpy.__version__)

from rag_fusion import load_documents, build_vectorstore, get_fusion_chain

st.title("🧠 Fusion RAG App (LangChain + Groq)")

uploaded_file = st.file_uploader("Upload a document (PDF/TXT)", type=["pdf", "txt"])
query = st.text_input("Ask a question")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    documents = load_documents(tmp_path)
    vectorstore = build_vectorstore(documents)
    chain = get_fusion_chain(vectorstore)

    if query:
        with st.spinner("Answering using Fusion RAG..."):
            result = chain.invoke({"query": query})

            st.markdown("### 💬 Answer")
            st.write(result["result"])

            st.markdown("### 📄 Sources")
            for doc in result["source_documents"]:
                st.write(doc.page_content[:300] + "...")
