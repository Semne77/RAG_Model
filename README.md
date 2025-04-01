# 🎾 Tennis Q&A — RAG Pipeline

This project is a small **Retrieval-Augmented Generation (RAG) application** built to answer tennis-related questions using a local knowledge base of `.txt` documents.

It demonstrates how to combine **document search** and **AI text generation** using Python, FAISS, and HuggingFace models.

---

## 🚀 What this program does

✅ Splits long tennis articles into small, meaningful text chunks  
✅ Converts these chunks into vector embeddings  
✅ Stores them in a FAISS vector database for fast similarity search  
✅ When the user asks a question:
- It finds the most relevant chunks
- It passes the context to a pre-trained language model (`Flan-T5`)  
✅ Returns a final, AI-generated answer

---

## 📂 Project Structure
```text
Tennis-RAG-App/
├── Code/
│   ├── version2.py      # Terminal interface version
│   └── app.py           # GUI application version
├── data/
│   ├── doc_file1.txt    # Sample tennis document
│   ├── doc_file2.txt    # Additional knowledge files
│   └── doc_file3.txt
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