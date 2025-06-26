import logging
import pandas as pd
from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import glob
import asyncio
import PyPDF2

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[logging.FileHandler(os.path.join(log_dir, "app.log"), encoding='utf-8'),
              logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("gemini_api_key")
pdf_folder = "data"
vector_store_dir = "vector_store"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "gemini-2.0-flash"

# Validate environment variables and file paths
if not api_key:
    logger.error("Google API key not found in environment variables.")
    raise ValueError("Google API key not found. Set 'gemini_api_key' in .env file.")

if not os.path.exists(pdf_folder):
    logger.error(f"Document folder not found at: {pdf_folder}")
    raise FileNotFoundError(f"Document folder not found at: {pdf_folder}")

# Create vector store directory
os.makedirs(vector_store_dir, exist_ok=True)

# Agent State
class AgentState(TypedDict):
    docs: List[Document]
    split_docs: List[Document]
    embeddings: any
    vector_db: any
    retriever: any
    llm: any
    qa_chain: any
    messages: Annotated[list, add_messages]
    error: Optional[str]

# Function to read PDF files
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
        logger.error(f"Failed to read PDF {file_path}: {e}")
        return ""

# Node Functions
async def load_documents(state: AgentState):
    logger.info("Loading documents from data directory...")
    try:
        pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        if not pdf_files:
            logger.error("No PDF files found in the data directory.")
            return {"error": "No PDF files found in the data directory."}

        logger.info(f"Found {len(pdf_files)} PDF files. Reading them...")

        docs = []
        for pdf_file in pdf_files:
            full_text = await asyncio.to_thread(read_pdf, pdf_file)
            if not full_text.strip():
                logger.error(f"No text found in PDF file: {pdf_file}")
                continue
            doc = Document(page_content=full_text, metadata={"source": pdf_file})
            docs.append(doc)

        if not docs:
            return {"error": "No valid documents were loaded from PDF files."}

        logger.info(f"Loaded {len(docs)} documents.")
        return {"docs": docs}
    except Exception as e:
        logger.error(f"Error loading PDF document: {e}")
        return {"error": f"Failed to load PDF document: {e}"}

async def split_text(state: AgentState):
    logger.info("Splitting text into chunks...")
    try:
        if not state.get("docs"):
            logger.error("No documents to split.")
            return {"error": "No documents to split."}

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(state["docs"])
        logger.info(f"Split into {len(split_docs)} document chunks.")
        return {"split_docs": split_docs}
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return {"error": f"Failed to split text: {e}"}

async def setup_embeddings(state: AgentState):
    logger.info("Initializing embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        logger.info("Embeddings initialized.")
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        return {"error": f"Failed to initialize embeddings: {e}"}

async def embed_docs(state: AgentState):
    logger.info("Embedding documents...")
    try:
        if not state.get("split_docs"):
            logger.error("No documents to embed.")
            return {"error": "No documents to embed."}

        embeddings = state["embeddings"]
        if not embeddings:
            logger.error("Embeddings not initialized.")
            return {"error": "Embeddings not initialized."}

        # Check if vector store already exists
        if os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
            logger.info(f"Loading existing vector store from {vector_store_dir}...")
            state["vector_db"] = FAISS.load_local(
                vector_store_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("Creating new vector store...")
            state["vector_db"] = FAISS.from_documents(state["split_docs"], embeddings)
            state["vector_db"].save_local(vector_store_dir)
            logger.info(f"Vector store saved to {vector_store_dir}.")

        logger.info("Documents embedded successfully.")
        return {"vector_db": state["vector_db"]}
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        return {"error": f"Failed to embed documents: {e}"}

async def setup_retriever(state: AgentState):
    logger.info("Setting up retriever...")
    try:
        retriever = state["vector_db"].as_retriever(search_kwargs={"k": 4})
        logger.info("Retriever setup completed.")
        return {"retriever": retriever}
    except Exception as e:
        logger.error(f"Error setting up retriever: {e}")
        return {"error": f"Failed to set up retriever: {e}"}

async def setup_llm(state: AgentState):
    logger.info("Initializing LLM...")
    try:
        llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key)
        logger.info("LLM initialized.")
        return {"llm": llm}
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return {"error": f"Failed to initialize LLM: {e}"}

async def build_chain(state: AgentState):
    logger.info("Building QA chain...")
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a support assistant for eBay users. Answer exclusively from the eBay User Agreement provided as document context, ensuring responses are accurate and policy-compliant. For procedural questions, provide concise step-by-step instructions. Do not rely on external knowledge. Accept queries in Hindi, English, or Hinglish, and respond in the same language as the query. Highlight arbitration processes when relevant to dispute-related queries."""),
            ("human", "{question}")
        ])
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=state["llm"],
            retriever=state["retriever"],
            memory=memory,
            condense_question_prompt=prompt_template,
            return_source_documents=True,
            output_key="answer"
        )
        logger.info("QA chain built successfully.")
        return {"qa_chain": qa_chain}
    except Exception as e:
        logger.error(f"Error building QA chain: {e}")
        return {"error": f"Failed to build QA chain: {e}"}

async def qa_agent(state: AgentState):
    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages found in state")
        return {"messages": messages, "error": "No question provided"}
    
    question = messages[-1].content
    logger.info(f"Received user question: {question}")
    
    try:
        response = await state["qa_chain"].ainvoke({
            "question": question,
            "chat_history": []
        })
        answer = response["answer"]
        retrieved_docs = response.get("source_documents", [])
        logger.info(f"Retrieved {len(retrieved_docs)} documents for question: {question}")
        for i, doc in enumerate(retrieved_docs):
            doc_content = doc.page_content[:200].encode('ascii', errors='ignore').decode('ascii')
            logger.info(f"Document {i+1}: {doc_content}...")
        logger.info(f"Generated answer: {answer}")
        return {"messages": messages + [SystemMessage(content=answer)]}
    except Exception as e:
        logger.error(f"Error in QA chain: {e}")
        return {"messages": messages, "error": f"Failed to process question: {e}"}

def should_run_qa_agent(state: AgentState):
    return "qa_agent" if state.get("messages") else END

# Compile LangGraph
def compile_graph():
    builder = StateGraph(AgentState)
    builder.add_node("load_documents", load_documents)
    builder.add_node("split_text", split_text)
    builder.add_node("setup_embeddings", setup_embeddings)
    builder.add_node("embed_docs", embed_docs)
    builder.add_node("setup_retriever", setup_retriever)
    builder.add_node("setup_llm", setup_llm)
    builder.add_node("build_chain", build_chain)
    builder.add_node("qa_agent", qa_agent)

    builder.set_entry_point("load_documents")
    builder.add_edge("load_documents", "split_text")
    builder.add_edge("split_text", "setup_embeddings")
    builder.add_edge("setup_embeddings", "embed_docs")
    builder.add_edge("embed_docs", "setup_retriever")
    builder.add_edge("setup_retriever", "setup_llm")
    builder.add_edge("setup_llm", "build_chain")
    builder.add_conditional_edges("build_chain", should_run_qa_agent, {"qa_agent": "qa_agent", END: END})
    builder.add_edge("qa_agent", END)

    return builder.compile()

# Global graph instance
graph = compile_graph()

# Main execution
async def run_graph(question: str):
    if not question or not question.strip():
        logger.warning("Empty or invalid question provided.")
        return {"error": "Please enter a valid question."}
    
    logger.info(f"Processing question: {question}")
    
    state = {
        "docs": [],
        "split_docs": [],
        "embeddings": None,
        "vector_db": None,
        "retriever": None,
        "llm": None,
        "qa_chain": None,
        "messages": [HumanMessage(content=question)],
        "error": None
    }
    
    try:
        result = await graph.ainvoke(state)
        error = result.get("error", None)
        
        if error:
            logger.error(f"Graph returned error: {error}")
            return {"error": error}
        
        stream = result.get("stream")
        if stream:
            async for output in stream():
                return {
                    "answer": output["answer"],
                    "source_documents": output["source_documents"],
                    "num_chunks": result.get("num_chunks", 0)
                }
        else:
            logger.warning("No stream found in graph result.")
            return {"error": "No answer generated. Please try again."}
    except Exception as e:
        logger.error(f"Error running graph: {e}")
        return {"error": f"An error occurred: {e}"}

if __name__ == "__main__":
    asyncio.run(run_graph())