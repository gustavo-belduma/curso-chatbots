"""
Created by Datoscout on 12/02/2025
solutions@datoscout.ec

This script implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain.
It loads a text file, splits it into smaller chunks, converts it into vector embeddings,
stores it in a FAISS vector database, and then queries the database using an LLM.
"""

# Standard imports
import os  # Handles file paths and environment variables

# Third party imports
import langchain  # Main framework for building LLM-based applications

# Callback handler to display intermediate results
from langchain.callbacks import StdOutCallbackHandler

# LangChain modules for question-answering and LLM integration
from langchain.chains import RetrievalQA  # Constructs a RAG pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into chunks

# LangChain modules for document processing, embeddings, and vector storage
from langchain_community.document_loaders import TextLoader  # Loads text documents
from langchain_community.vectorstores import FAISS  # Vector database for efficient search
from langchain_openai import ChatOpenAI  # OpenAI's LLM integration
from langchain_openai import OpenAIEmbeddings  # Converts text into vector embeddings

# Internal imports
from src.config.settings import (
    DATA_PATH,
    OPENAI_API_KEY,
    OPENAI_COMPLETIONS_MODEL,
    OPENAI_EMBEDDINGS_MODEL,
)

print(f"LangChain version: {langchain.__version__}")

# -------------------------------
# 1. Load text file
# -------------------------------

# Define the path to the text file (Alice in Wonderland)
text = os.path.join(DATA_PATH, "Alicia en el país de las maravillas.txt")

# Load the text file using LangChain's TextLoader
loader = TextLoader(file_path=text, encoding="utf-8")

# -------------------------------
# 2. Split text into smaller chunks
# -------------------------------

# This helps improve retrieval efficiency and accuracy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Each chunk will have 500 characters
    chunk_overlap=0,  # No overlap between consecutive chunks
)

# Apply text splitting to the loaded document
data = loader.load_and_split(text_splitter=text_splitter)

# Display a small portion of the data (for debugging)
data[5]

# --------------------------------------
# 3. Convert text chunks into embeddings
# --------------------------------------

# OpenAI Embeddings model converts text into numerical vectors
embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDINGS_MODEL,
    api_key=OPENAI_API_KEY,  # API key for OpenAI access
    show_progress_bar=True,  # Displays progress while embedding
)

# -------------------------------
# 4. Store embeddings in FAISS vector database
# -------------------------------

# FAISS (Facebook AI Similarity Search) stores and retrieves similar embeddings efficiently
index = FAISS.from_documents(data, embeddings)

# -------------------------------
# 5. Perform a similarity search
# -------------------------------

# Define a query for the chatbot
query = "¿es buena idea retroceder al día de ayer?"

# Perform a search in the FAISS index for similar text chunks
best_docs = index.similarity_search_with_relevance_scores(query, k=2)
# best_docs_2 = index.similarity_search(query, k=2)
print(best_docs)
# print(best_docs_2)

# ------------------------------------------------------
# 6. Configure the retriever (tuning search parameters)
# ------------------------------------------------------

retriever = index.as_retriever()

# Adjust search parameters for better retrieval:
retriever.search_kwargs["fetch_k"] = 10  # Fetch 10 documents before re-ranking
retriever.search_kwargs["maximal_marginal_relevance"] = True  # Ensures diverse results
retriever.search_kwargs["k"] = 3  # Return top 3 most relevant documents

# -------------------------------
# 7. Set up the LLM for answering questions
# -------------------------------

# Initialize the OpenAI language model (GPT-based)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_COMPLETIONS_MODEL)

# -------------------------------
# 8. Build the RAG pipeline
# -------------------------------

# RetrievalQA combines the retriever (search) with the LLM (answer generation)
chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, verbose=False  # Enables logging for debugging
)

# -------------------------------
# 9. Enable real-time output display
# -------------------------------

# The callback handler prints intermediate results to the console
handler = StdOutCallbackHandler()

# -------------------------------
# 12. Query the chatbot
# -------------------------------

# First query
response = chain.invoke(
    {"query": query}, config={"callbacks": [handler]}  # Enables real-time output
)
print(response["result"])  # Display the chatbot's response

# # Second query (another test question)
query = "What path should you take if you don't know where you are going?"
response = chain.invoke({"query": query}, config={"callbacks": [handler]})
print(response["result"])  # Display the chatbot's response
