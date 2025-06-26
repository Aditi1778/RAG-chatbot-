import os
from dotenv import load_dotenv
import logging
import asyncio
import glob
import pandas as pd
from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2
import spacy
import json
import re

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[logging.FileHandler(os.path.join(log_dir, "app.log"), encoding='utf-8'),
              logging.StreamHandler()]
)

nlp = spacy.load("en_core_web_sm")

# Function to read PDF resume
def read_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        return text
    except Exception as e:
        logging.error(f"Failed to read PDF {file_path}: {e}")
        return ""

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_GENAI_API_KEY")
data_dir = "data"
vector_store_dir = "vector_store"  # Directory to save the vector store
os.makedirs(vector_store_dir, exist_ok=True)  # Create vector store directory if it doesn't exist

if not os.path.exists(data_dir):
    logging.error(f"Data directory '{data_dir}' does not exist.")
    raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

class AgentState(TypedDict):
    query: str
    chunks: List[str]
    response: str
    docs: List[str]
    embeddings: Any
    vector_db: Any
    prompt: str
    model: str
    error: Optional[str]

async def load_documents(state: AgentState):
    logging.info("Loading documents from data directory...")
    try:
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
        if not pdf_files:
            logging.error("No PDF files found in the data directory.")
            return {"error": "No PDF files found in the data directory."}

        logging.info(f"Found {len(pdf_files)} PDF files. Reading them...")

        docs = []
        for pdf_file in pdf_files:
            full_text = await asyncio.to_thread(read_pdf, pdf_file)

            if not full_text.strip():
                logging.error(f"No text found in PDF file: {pdf_file}")
                continue

            doc = Document(page_content=full_text, metadata={"source": pdf_file})
            docs.append(doc)

        if not docs:
            return {"error": "No valid documents were loaded from PDF files."}

        logging.info(f"Loaded {len(docs)} documents.")
        return {"docs": docs}

    except Exception as e:
        logging.error(f"Error loading PDF document: {e}")
        return {"error": f"Failed to load PDF document: {e}"}

async def split_text(state: AgentState):
    logging.info("Splitting text into chunks...")
    try:
        if not state.get("docs"):
            logging.error("No documents to split.")
            return {"error": "No documents to split."}

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(state["docs"])
        logging.info(f"Split into {len(docs)} documents.")
        return {"docs": docs}
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        return {"error": f"Failed to split text: {e}"}

async def embed_docs(state: AgentState):
    try:
        logging.info("Embedding documents...")
        if not state.get("docs"):
            logging.error("No documents to embed.")
            return {"error": "No documents to embed."}

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        state["embeddings"] = embeddings

        # Check if vector store already exists
        if os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
            logging.info(f"Loading existing vector store from {vector_store_dir}...")
            state["vector_db"] = FAISS.load_local(
                vector_store_dir,
                embeddings,
                allow_dangerous_deserialization=True  # Required for loading FAISS index
            )
        else:
            logging.info("Creating new vector store...")
            state["vector_db"] = FAISS.from_documents(state["docs"], embeddings)
            # Save the vector store to disk
            state["vector_db"].save_local(vector_store_dir)
            logging.info(f"Vector store saved to {vector_store_dir}.")

        logging.info("Documents embedded successfully.")
        return {"vector_db": state["vector_db"]}

    except Exception as e:
        logging.error(f"Error embedding documents: {e}")
        return {"error": f"Failed to embed documents: {e}"}

# Define the workflow
workflow = StateGraph(AgentState)

workflow.add_node("load_documents", load_documents)
workflow.add_node("split_text", split_text)
workflow.add_node("embed_docs", embed_docs)

workflow.add_edge("load_documents", "split_text")
workflow.add_edge("split_text", "embed_docs")
workflow.add_edge("embed_docs", END)

workflow.set_entry_point("load_documents")

graph = workflow.compile()

async def run_knowledge_base():
    initial_state = {
        "docs": [],
        "embeddings": None,
        "error": ""
    }

    try:
        final_state = await graph.ainvoke(initial_state)

        if final_state.get("error"):
            logging.error(f"Workflow failed: {final_state['error']}")
            print(f"Error: {final_state['error']}")
        else:
            logging.info("Knowledge base creation completed successfully.")
            print(f"Loaded {len(final_state['docs'])} document chunks.")
            print(f"Vector store created and saved to {vector_store_dir}.")

    except Exception as e:
        logging.error(f"Workflow execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_knowledge_base())