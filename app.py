import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Streamlit app configuration
st.set_page_config('PDFChatBot')
st.header("Ask questions about your PDF")

# Initialize global embeddings model
if 'embedding_model' not in st.session_state: 
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize global knowledge base
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

# Preinitialize the LLM
if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name='gpt-3.5-turbo')

# This chain coordinates the LLM for the specific QA task
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = load_qa_chain(st.session_state.llm, chain_type="stuff")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF", type="pdf", on_change=st.cache_resource.clear)

@st.cache_resource
def create_knowledge_base(pdf):
    # Read the PDF and extract text
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)

    # Filter content: remove blank lines
    text = "\n".join(line for line in text.split("\n") if line.strip())

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Chunk size
        chunk_overlap=100,  # Overlap between chunks
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings using the specified model from HuggingFace
    knowledge_base = FAISS.from_texts(chunks, st.session_state.embedding_model)

    return knowledge_base

# If a PDF file is uploaded
if pdf_file:
    st.session_state.knowledge_base = create_knowledge_base(pdf_file)
    user_question = st.text_area("Ask something about your PDF:")

    if user_question:
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        relevant_docs = st.session_state.knowledge_base.similarity_search(user_question, 3)

        # Uncomment if you want to see "relevant_docs" content
        # st.write("Array of relevant documents:")
        # st.write(relevant_docs)

        answer = st.session_state.qa_chain.invoke(input={"input_documents": relevant_docs, "question": user_question})        
        st.write(answer['output_text'])
