# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# --------------------------
# Load .env
# --------------------------
load_dotenv()

# --------------------------
# Initialize session state
# --------------------------
def init_session_state():
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = os.getenv("QDRANT_URL")
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

# --------------------------
# Process PDF and create embeddings
# --------------------------
def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --------------------------
# Initialize Qdrant vector store
# --------------------------
def init_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.session_state.openai_api_key
    )

    vector_db = Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key,
        collection_name="legal_documents"
    )
    return vector_db

# --------------------------
# Main App
# --------------------------
def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Document Analyzer ⚖️")

    # Sidebar: API keys
    with st.sidebar:
        st.header("🔑 API Configuration")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key or ""
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url or "",
            help="Local: http://localhost:6333 | Cloud: your Qdrant Cloud URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url

        qdrant_key = st.text_input(
            "Qdrant API Key (optional for local)",
            type="password",
            value=st.session_state.qdrant_api_key or ""
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key

    # Upload PDF
    uploaded_file = st.file_uploader("Upload Legal PDF", type=["pdf"])
    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.spinner("Processing PDF and creating embeddings..."):
            chunks = process_pdf(uploaded_file)
            st.session_state.vector_db = init_vectorstore(chunks)
            st.session_state.processed_files.add(uploaded_file.name)
            st.success("✅ Document processed and embeddings created!")

    if not st.session_state.vector_db:
        st.info("Please upload a PDF and configure API keys to start analysis.")
        return

    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Contract Review", "Legal Research", "Risk Assessment", "Compliance Check", "Custom Query"]
    )

    user_query = None
    if analysis_type == "Custom Query":
        user_query = st.text_area("Enter your specific query:")

    if st.button("Analyze"):
        if analysis_type == "Custom Query" and not user_query:
            st.warning("Please enter a query")
        else:
            query_text = ""
            if analysis_type != "Custom Query":
                predefined_tasks = {
                    "Contract Review": "Review this contract and identify key terms, obligations, and potential issues.",
                    "Legal Research": "Research relevant cases and precedents related to this document.",
                    "Risk Assessment": "Analyze potential legal risks and liabilities in this document.",
                    "Compliance Check": "Check this document for regulatory compliance issues."
                }
                query_text = predefined_tasks[analysis_type]
            else:
                query_text = user_query

            # Use LangChain RetrievalQA
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model_name="gpt-4",
                    openai_api_key=st.session_state.openai_api_key
                ),
                chain_type="stuff",
                retriever=st.session_state.vector_db.as_retriever(),
                return_source_documents=True
            )

            with st.spinner("Analyzing document..."):
                result = qa({"query": query_text})
                answer = result["result"]
                sources = result["source_documents"]

                st.header("📄 Analysis Result")
                st.markdown(answer)

                st.header("📑 Source References")
                for doc in sources:
                    st.markdown(f"- {doc.metadata.get('source', 'Unknown page')}")

if __name__ == "__main__":
    main()
