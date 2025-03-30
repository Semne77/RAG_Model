from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Constants and setup
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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50
)

# Load documents and prepare chunks
def load_documents_and_build_index(data_folder="data"):
    documents = {}
    all_chunks = []
    metadata = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            path = os.path.join(data_folder, filename)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                doc_id = f"doc_{filename.split('.')[0]}"
                documents[doc_id] = text
                print(f"✅ Loaded {doc_id}, {len(text.split())} words")

    for doc_id, text in documents.items():
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        for i, chunk in enumerate(chunks):
            metadata.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk
            })

    # Embed and index
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, metadata, all_chunks

# Optional numeric fact extractor
def extract_numeric_facts(chunks, query):
    pattern = r'(\w+\s\w+).*?(\d+)\s.*?(Grand Slam|major|Wimbledon|US Open|Roland Garros|French Open|Australian Open).*?titles'
    query = query.lower()
    query_number = next((int(num) for num in re.findall(r'\d+', query)), None)

    for chunk in chunks:
        match = re.search(pattern, chunk, re.IGNORECASE)
        if match:
            player = match.group(1)
            num = int(match.group(2))
            tournament = match.group(3).lower()
            if query_number == num and tournament in query:
                return f"✅ Answer from fact: {player}"
    return None

# Main RAG function
def ask_question(query):
    index, metadata, all_chunks = load_documents_and_build_index()

    # Retrieve
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k=8)
    top_k_chunks = [metadata[idx]['text'] for idx in I[0]]

    print("\n📌 Top Matching Chunks:")
    for rank, chunk_text in enumerate(top_k_chunks):
        print(f"\n#{rank+1}:\n📄 {chunk_text}")

    # Check for direct numeric match
    fact_match = extract_numeric_facts(top_k_chunks, query)
    if fact_match:
        print("\n🧠 Using numeric match...")
        print("🧠 Final Answer:\n", fact_match)
        return

    # Otherwise, use generator
    max_input_tokens = 1024
    context = ""
    total_tokens = 0
    for chunk in top_k_chunks:
        tokens = tokenizer.encode(chunk, add_special_tokens=False, max_length=1024, truncation=True)
        if total_tokens + len(tokens) > max_input_tokens:
            break
        context += chunk + "\n\n"
        total_tokens += len(tokens)

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

if __name__ == "__main__":
    while True:
        query = input("\n🔍 Ask a Question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        ask_question(query)