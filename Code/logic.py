"""
RAG Pipeline - Question Answering System

This script implements a simple Retrieval-Augmented Generation (RAG) pipeline.
It uses FAISS for semantic search, SentenceTransformers for embedding, 
and Google's FLAN-T5 model for text generation.

Features:
- Loads `.txt` documents from the local 'data/' folder.
- Splits documents into smaller chunks for better retrieval.
- Embeds and indexes the chunks using FAISS.
- Retrieves top-k relevant chunks based on the user's query.
- Optionally extracts numeric facts if present in the retrieved chunks.
- Uses FLAN-T5 to generate an answer based on the retrieved context.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# Environment Setup
# =========================
load_dotenv()  # Load environment variables from .env file

# =========================
# Model & Tokenizer Setup
# =========================
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 1024
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Use GPU if available
)

embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# =========================
# Text Splitter for Chunking
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50
)

# =========================
# Functions
# =========================

def load_documents_and_build_index(data_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")):
    """
    Loads all text documents from the data folder,
    splits them into chunks, embeds them, and builds a FAISS index.

    Args:
        data_folder (str): Path to the folder containing text files.

    Returns:
        index (faiss.IndexFlatIP): FAISS index for similarity search.
        metadata (list): Metadata about each text chunk.
        all_chunks (list): List of all text chunks.
    """
    documents = {}
    all_chunks = []
    metadata = []

    # Read all .txt documents
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            path = os.path.join(data_folder, filename)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                doc_id = f"doc_{filename.split('.')[0]}"
                documents[doc_id] = text
                print(f"✅ Loaded {doc_id}, {len(text.split())} words")

    # Split documents into chunks
    for doc_id, text in documents.items():
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        for i, chunk in enumerate(chunks):
            metadata.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk
            })

    # Embed and build FAISS index
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, metadata, all_chunks

def ask_question(query):
    """
    Handles the full RAG pipeline:
    1. Loads documents & builds FAISS index.
    2. Retrieves top-k matching chunks.
    3. Uses FLAN-T5 to generate answer.

    Args:
        query (str): User's question.

    Returns:
        str: Final answer to the question.
    """
    index, metadata, all_chunks = load_documents_and_build_index()

    # Step 1: Retrieve relevant chunks
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k=8)
    top_k_chunks = [metadata[idx]['text'] for idx in I[0]]

    print("\n📌 Top Matching Chunks:")
    for rank, chunk_text in enumerate(top_k_chunks):
        print(f"\n#{rank+1}:\n📄 {chunk_text}")

    # Step 2: Context for generator
    max_input_tokens = 1024
    context = ""
    total_tokens = 0
    for chunk in top_k_chunks:
        tokens = tokenizer.encode(chunk, add_special_tokens=False, max_length=1024, truncation=True)
        if total_tokens + len(tokens) > max_input_tokens:
            break
        context += chunk + "\n\n"
        total_tokens += len(tokens)

    # Step 3: Prompt & generate answer
    prompt = f"""Answer the following question using the context below:

Context:
{context}

Question: {query}
"""
    print("\n🧠 Generating answer with FLAN...\n")
    response = generator(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
    final_answer = response.strip()
    print("🧠 Final Answer:\n", final_answer)
    return final_answer

# =========================
# CLI Interface
# =========================
if __name__ == "__main__":
    while True:
        query = input("\n🔍 Ask a Question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        ask_question(query)
