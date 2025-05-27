import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA  
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def load_documents(file_path):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    return TextLoader(file_path).load()

from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vectorstore(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"} 
    )

    return FAISS.from_documents(chunks, embeddings)


def get_fusion_chain(vectorstore):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    fusion_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return fusion_chain

