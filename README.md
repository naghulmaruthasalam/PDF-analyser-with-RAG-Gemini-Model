# PDF-analyser-with-RAG-Gemini-Model
A lightweight PDF Q&amp;A system using Retrieval-Augmented Generation (RAG). It extracts text, splits into chunks, generates embeddings with Gemini, and uses FAISS for fast similarity search. Retrieves relevant context and produces accurate answers via LLM, demonstrating end-to-end RAG pipeline implementation.
🧠 Overview

This project implements a simple end-to-end RAG pipeline:

Extracts text from PDF documents
Splits content into manageable chunks
Converts chunks into vector embeddings using Gemini
Stores embeddings in FAISS for efficient similarity search
Retrieves relevant chunks based on user queries
Generates context-aware answers using a Gemini LLM
⚙️ Tech Stack
Python
Streamlit – UI for interaction
FAISS – Vector similarity search
Google Gemini API – Embeddings & LLM
PyPDF – PDF text extraction
NumPy – Numerical operations
🔄 How It Works
Upload a PDF file
Extract and preprocess text
Split text into chunks
Generate embeddings for each chunk
Store embeddings in FAISS index
Convert user query into embedding
Retrieve top-k similar chunks
Pass context to Gemini for response generation
✨ Features
Simple and intuitive UI using Streamlit
Fast similarity search using FAISS
Context-aware question answering
Handles large PDF documents
Modular and easy to extend
🚧 Future Improvements
Add chat history (memory)
Implement hybrid search (keyword + vector)
Improve chunking strategies
Add API backend using FastAPI
Optimize performance with batching and caching
