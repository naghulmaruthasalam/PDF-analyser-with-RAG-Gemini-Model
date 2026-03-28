import streamlit as st
import os
import numpy as np
import faiss
import google.generativeai as genai
from pypdf import PdfReader


genai.configure(api_key='Your key')

st.title("📄 PDF Analyzer (RAG with Gemini)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_embeddings(texts):
    embeddings = []
    for t in texts:
        emb = genai.embed_content(
            model="models/gemini-embedding-001", 
            content=t
        )
        embeddings.append(emb["embedding"])
    return np.array(embeddings).astype("float32")

if uploaded_file:
    text = read_pdf(uploaded_file)

    if not text.strip():
        st.error("No readable text found in PDF")
    else:
        chunks = chunk_text(text)

        st.write(f"Total chunks: {len(chunks)}")

        # Create embeddings
        embeddings = get_embeddings(chunks)

        # Store in FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.success("PDF processed successfully!")

        query = st.text_input("Ask something about the PDF")

        if query:
            # Embed query
            query_embedding = genai.embed_content(
                model="models/gemini-embedding-001",
                content=query
            )["embedding"]

            query_embedding = np.array([query_embedding]).astype("float32")

            # Similarity search
            k = 3
            distances, indices = index.search(query_embedding, k)

            retrieved_chunks = [chunks[i] for i in indices[0]]
            context = "\n".join(retrieved_chunks)

            # Generate response
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
            Answer the question based ONLY on the context below.

            Context:
            {context}

            Question:
            {query}
            """

            response = model.generate_content(prompt)

            st.write("### Answer:")
            st.write(response.text)