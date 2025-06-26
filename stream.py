import streamlit as st
import asyncio
import logging
import os
from app import run_graph, llm_model

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[logging.FileHandler(os.path.join(log_dir, "streamlit.log"), encoding='utf-8'),
              logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Streamlit UI configuration
st.set_page_config(page_title="eBay Support Chatbot", layout="wide")
st.title("eBay Support Chatbot")
st.markdown("Ask questions about the eBay User Agreement. Responses are generated based on the provided document.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# Sidebar
with st.sidebar:
    st.header("Chatbot Information")
    st.markdown(f"**Current Model:** {llm_model}")
    st.markdown(f"**Indexed Document Chunks:** {st.session_state.num_chunks}")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.num_chunks = 0
        st.success("Chat history cleared!")
        logger.info("Chat history cleared by user.")

# Input form for natural language queries
with st.form(key="question_form"):
    question = st.text_input("Your Question (in English, Hindi, or Hinglish):", placeholder="e.g., How do I resolve a dispute on eBay?")
    submit_button = st.form_submit_button("Submit")

# Handle form submission with streaming
if submit_button and question:
    logger.info(f"User submitted question: {question}")
    
    # Placeholder for streaming response
    response_container = st.empty()
    with response_container.container():
        st.markdown("### Answer")
        answer_placeholder = st.empty()
    
    # Run the graph and process response
    result = asyncio.run(run_graph(question))
    
    if "error" in result:
        st.error(result["error"])
        logger.error(f"Error in response: {result['error']}")
    else:
        answer = result["answer"]
        source_documents = result.get("source_documents", [])
        st.session_state.num_chunks = result.get("num_chunks", st.session_state.num_chunks)
        
        # Stream the answer character by character
        def stream_answer():
            for char in answer:
                yield char
        
        with answer_placeholder:
            st.write_stream(stream_answer())
        
        # Add to chat history
        st.session_state.chat_history.append({"question": question, "answer": answer})
        
        # Display source text passages
        if source_documents:
            with st.expander("Source Text Passages"):
                for i, doc in enumerate(source_documents):
                    doc_content = doc.page_content[:200].encode('ascii', errors='ignore').decode('ascii')
                    st.markdown(f"**Document {i+1} ({doc.metadata.get('source', 'Unknown')}):**")
                    st.write(f"{doc_content}...")
                    logger.info(f"Displayed document {i+1}: {doc_content}...")

# Display chat history
if st.session_state.chat_history:
    with st.expander("Chat History"):
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            st.markdown(f"**A{i+1}:** {chat['answer']}")
            st.markdown("---")

# Footer with instructions
st.markdown("""
---
**Instructions:**
- Enter your question in English, Hindi, or Hinglish.
- The chatbot answers based on the eBay User Agreement.
- For dispute-related queries, arbitration details will be highlighted.
- View source text passages and chat history by expanding the sections below the answer.
- Use the sidebar to clear chat history or view model and document information.
""")