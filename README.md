# 🎾 Tennis Q&A — RAG Pipeline


![Tennis Banner](image/Big3.jpg)

This project is a small **Retrieval-Augmented Generation (RAG) application** built to answer questions about Tennis "Gig 3" (Novak Djokovic, Roger Federer, Rafael Nadal) using a local knowledge base of `.txt` documents.

It demonstrates how to combine **document search** and **AI text generation** using Python, FAISS, and HuggingFace models.

---

## 🚀 What this program does

✅ Splits long tennis articles into small, meaningful text chunks  
✅ Converts these chunks into vector embeddings  
✅ Stores them in a FAISS vector database for fast similarity search  
✅ When the user asks a question:
- It finds the most relevant chunks
- It passes the context to a pre-trained language model (`Flan-T5`)  
- ✅ Returns a final, AI-generated answer

---

## 📂 Project Structure
```text
Tennis-RAG-App/
├── Code/
│   ├── logic.py      # Terminal interface version
│   └── gui_app.py           # GUI application version
├── data/
│   ├── doc_file1.txt    # Sample tennis document
│   ├── doc_file2.txt    # Additional knowledge files
│   └── doc_file3.txt
│── image/
│   └── Big3.jpg
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 Installation & Running

### 1. Clone the Repository

```bash
git clone https://github.com/Semne77/Transformer.git
cd Transformer
```
### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies to run the project
```bash
pip install -r requirements.txt
```


### 4. Run the program
```bash
cd Code
python gui_app.py
```

## Sample Questions that you may ask:

```text
- Which player is from Spain?

- Which player knows English, German and French languages?

- Which player won all nine ATP Masters 1000 tournaments?
```

**Note:** This project uses the Flan-T5 model from Hugging Face. 
To run the project, you must:
1. Create a free Hugging Face account: https://huggingface.co/join
2. Accept the model license agreement here: https://huggingface.co/google/flan-t5-base
3. Add your Hugging Face API token to your environment:
   ```bash
   export HUGGINGFACEHUB_API_TOKEN=your_token_here