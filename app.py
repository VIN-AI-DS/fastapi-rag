import os
import faiss
import torch
import pickle
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from llama_cpp import Llama  # Runs Llama-2 locally
from langchain_groq import ChatGroq

# Load Llama-2 model (Replace with Mistral if needed)
llm = ChatGroq(
    model="mistral-saba-24b",
    api_key=os.environ['GROQ_API_KEY']
)
# Function to load and process any document type
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)

    docs = loader.load()

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    return chunks

# Function to create FAISS index
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

# Load or create vectorstore
def load_or_create_vectorstore(file_path):
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
    else:
        chunks = load_document(file_path)
        vectorstore = create_vectorstore(chunks)

    return vectorstore

# Function to answer queries using RAG
def answer_question(vectorstore, query):
    retriever = vectorstore.as_retriever(search_k=3)
    docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = llm.invoke(prompt)
    return response.content

# Example Usage
if __name__ == "__main__":
    file_path = "/content/Tredence Analytics Infinite-AI - Circular.pdf"  # Change to any document type (PDF, TXT, DOCX)
    vectorstore = load_or_create_vectorstore(file_path)

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = answer_question(vectorstore, query)
        print("\nAnswer:", response)
