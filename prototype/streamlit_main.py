import streamlit as st
import requests
import json

FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="LangGraph Agent", layout="wide")

# Tabs for different tools
tab1, tab2, tab3 = st.tabs(["🔍 Search Pinecone", "🌐 Web Search", "💬 Chat Agent"])

# --- Search Pinecone Tool ---
with tab1:
    st.header("🔍 Pinecone Document Search")
    query = st.text_input("Enter your query")
    year = st.text_input("Year (optional)")
    quarter = st.text_input("Quarter (optional)")
    top_k = st.slider("Top K Results", 1, 10, 5)

    if st.button("Search Pinecone"):
        response = requests.post(f"{FASTAPI_URL}/run-tool", json={
            "tool": "search_pinecone",
            "query": query,
            "year": year,
            "quarter": quarter,
            "top_k": top_k
        })
        result = response.json()
        st.subheader("Results")
        st.text(result.get("result", "No result returned"))

# --- Web Search Tool ---
with tab2:
    st.header("🌐 Web Search Tool")
    web_query = st.text_input("Enter your web search query")
    if st.button("Search Web"):
        response = requests.post(f"{FASTAPI_URL}/run-tool", json={
            "tool": "web_search",
            "query": web_query
        })
        result = response.json()
        st.subheader("Web Results")
        st.text(result.get("result", "No result returned"))

# --- Chat Interface ---
with tab3:
    st.header("💬 Chat with the Research Agent")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask the agent a question...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = requests.post(f"{FASTAPI_URL}/chat", json={
                "query": user_input,
                "chat_history": st.session_state.chat_history
            })
            data = response.json()
            agent_reply = data.get("response", "No response from agent.")
            st.session_state.chat_history.append({"role": "agent", "content": agent_reply})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    st.subheader("📥 Download Latest Report")
    if st.button("Download Markdown Report"):
        download_response = requests.get(f"{FASTAPI_URL}/download-report")
        if download_response.status_code == 200:
            st.download_button(
                label="📄 Download Markdown File",
                data=download_response.content,
                file_name="nvidia_report.md",
                mime="text/markdown"
            )
        else:
            st.error("Failed to download report.")