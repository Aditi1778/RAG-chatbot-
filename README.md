# RAG-chatbot

# 🧠 Fine-Tuned RAG Chatbot with Streaming Responses

This project demonstrates the implementation of a **Retrieval-Augmented Generation (RAG)** based chatbot using open-source LLMs, embeddings, vector databases, and a Streamlit interface with real-time streaming support.

---

## 📌 Project Architecture

```text
📁 data/        → Raw input document (PDF, DOCX, etc.)
📁 chunks/      → Preprocessed text chunks (100–300 words)
📁 vectordb/    → Vector database (FAISS/Chroma) of embeddings
📁 notebooks/   → Jupyter notebooks for preprocessing & testing
📁 src/         → Core modules: retriever, generator, utils
📄 app.py       → Streamlit chatbot interface with streaming
📄 requirements.txt
📄 README.md
📄 report.pdf   → Architecture, prompt logic, sample outputs
⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/rag-chatbot-amlgo.git
cd rag-chatbot-amlgo
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🗂️ Document Preprocessing & Chunking
Load raw PDF using PyMuPDF or DOCX using python-docx

Clean and split into 100–300 word chunks

Save to /chunks/chunks.json

python
Copy
Edit
from src.preprocess import chunk_text
chunk_text(input_path='data/terms.pdf', output_path='chunks/chunks.json')
🔎 Embeddings & Vector Store
Generate embeddings using all-MiniLM-L6-v2 or bge-small-en

Store in FAISS or Chroma vector store

python
Copy
Edit
from src.embeddings import build_vector_store
build_vector_store(chunks_path='chunks/chunks.json', db_path='vectordb/')
🧠 RAG Pipeline
Retrieve top-k semantic matches from vector DB

Inject retrieved content + user query into prompt

Generate answer using an open-source LLM (e.g., Mistral-7B, Zephyr, or LLaMA)

Prompt Template Example:

css
Copy
Edit
You are a helpful assistant. Answer the question using the context below:

{retrieved_chunks}

Question: {user_query}
💬 Streamlit Chatbot UI (Streaming Enabled)
Run the app:

bash
Copy
Edit
streamlit run app.py
Features:

User input box

Streaming response (token-by-token)

Display source chunks

Sidebar shows: model name, # of indexed chunks

Clear/reset button

🧪 Sample Queries
Query	Expected Behavior
"What is the refund policy?"	Shows clause from the document
"How is my personal data used?"	Pulls from privacy section
"Can I share my account?"	Pulls T&C section on usage restrictions

📸 Demo

📽️ Video Demo Link

📄 report.pdf Includes
Chunking logic & document structure

Embedding model and DB reasoning

Prompt format and examples

3–5 sample queries (with success/failure cases)

Notes on hallucination or slow response

🔗 Resources Used
Sentence Transformers (MiniLM, BGE-small)

FAISS / Chroma for vector search

LangChain (optional)

Open-source LLMs: Mistral, LLaMA, Zephyr

Streamlit for real-time UI

🧠 Author
Aditi Goyal
📧 goyaladiti1708@gmail.com
🔗 LinkedIn
