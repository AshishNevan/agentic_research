import streamlit as st
import os
from dotenv import load_dotenv
import base64

# Import custom agents
# from backend.snowflake_agent.sql_agent import SnowflakeAgent
# from backend.rag_agent import RAGAgent
# from backend.web_search_agent import WebSearchAgent


load_dotenv()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_document" not in st.session_state:
    st.session_state.research_document = None

# Page configuration
st.set_page_config(page_title="NVIDIA Research Assistant", layout="wide")

# Sidebar
with st.sidebar:
    st.title("NVIDIA Research Assistant")
    st.markdown("This assistant uses multiple agents to generate research documents based on NVIDIA data:")
    st.markdown("- **Snowflake Agent**: Analyzes NVIDIA valuation measures")
    st.markdown("- **RAG Agent**: Processes NVIDIA 10-K/Q reports (2022-2025)")
    st.markdown("- **Web Search Agent**: Retrieves latest information")
    
    # Add configuration options here
    model_option = st.selectbox(
        "Select LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
    )
    
    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.research_document = None
        st.experimental_rerun()

# Main chat interface
st.title("NVIDIA Research Assistant")

# Initialize agents and orchestrator
@st.cache_resource
def load_agents():
    # snowflake_agent = SnowflakeAgent(os.environ.get('SNOWFLAKE_URI'))
    # rag_agent = RAGAgent()
    # web_search_agent = WebSearchAgent()
    # orchestrator = ResearchOrchestrator(
    #     snowflake_agent=snowflake_agent,
    #     rag_agent=rag_agent,
    #     web_search_agent=web_search_agent,
    #     model_name=model_option
    # )
    orchestrator = None #remove this once you add the agents to the system
    return orchestrator

orchestrator = load_agents()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "is_markdown" in message and message["is_markdown"]:
            st.markdown(message["content"])
        else:
            st.write(message["content"])

# Function to create a download link for the markdown file
def get_download_link(markdown_text, filename="research_document.md"):
    """Generate a download link for the markdown text"""
    b64 = base64.b64encode(markdown_text.encode()).decode()
    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">Download Research Document</a>'
    return href

# Chat input
if prompt := st.chat_input("Ask a question about NVIDIA..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response with a spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("Generating research document..."):
            # Call the orchestrator to generate the research document
            research_document = orchestrator.generate_research(prompt)
            
            # Store the research document in session state
            st.session_state.research_document = research_document
            
            # Display the markdown content
            st.markdown(research_document)
            
            # Add download button for the research document
            st.markdown(get_download_link(research_document), unsafe_allow_html=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": research_document,
        "is_markdown": True
    })

# Display download button for the latest research document if it exists
if st.session_state.research_document:
    st.sidebar.markdown("### Download Latest Research")
    st.sidebar.markdown(get_download_link(st.session_state.research_document), unsafe_allow_html=True)
    
    # Option to download as PDF (requires additional libraries)
    st.sidebar.markdown("### Advanced Export Options")
    if st.sidebar.button("Generate PDF"):
        with st.sidebar.spinner("Generating PDF..."):
            # This is a placeholder for PDF generation functionality
            # You would need to implement this using libraries like weasyprint or pdfkit
            st.sidebar.success("PDF generation feature coming soon!")
