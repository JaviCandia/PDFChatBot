import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Streamlit app configuration
st.set_page_config('PDFChatBot')
st.header("Ask questions about your PDF")

# Initialize global embeddings model 
if 'embeddings' not in st.session_state: 
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize global knowledge base
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

# Upload a PDF file
pdf_obj = st.file_uploader("Upload your PDF", type="pdf", on_change=st.cache_resource.clear)

@st.cache_resource
def create_embeddings(pdf):
    # Read the PDF and extract text
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)

    # Filter content: remove blank lines
    text = "\n".join(line for line in text.split("\n") if line.strip())

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Chunk size
        chunk_overlap=100,  # Overlap between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings using the specified model from HuggingFace
    knowledge_base = FAISS.from_texts(chunks, st.session_state.embeddings)

    return knowledge_base

# If a PDF file is uploaded
if pdf_obj:
    st.session_state.knowledge_base = create_embeddings(pdf_obj)
    user_question = st.text_area("Ask something about your PDF:")

    if user_question:
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        docs = st.session_state.knowledge_base.similarity_search(user_question, 3)

        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=user_question)

        st.write(answer)