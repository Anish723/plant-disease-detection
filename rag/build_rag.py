from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load knowledge file
loader = TextLoader("rag/knowledge/plant_diseases.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# Save database
vectorstore.save_local("rag/vectorstore")

print("RAG knowledge index built successfully!")