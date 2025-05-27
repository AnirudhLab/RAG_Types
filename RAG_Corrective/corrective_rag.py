import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def load_documents(path):
    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()
    return TextLoader(path).load()

def build_vectorstore(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # avoids meta tensor errors
    )
    return FAISS.from_documents(chunks, embeddings)

def get_corrective_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an intelligent assistant. First, try to answer the user's question using only the provided context.
If the context is insufficient or unclear, say what is missing and request clarification or additional retrieval.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain
