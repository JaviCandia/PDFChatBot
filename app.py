import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Streamlit app configuration
st.set_page_config('CV Scanner MVP')
st.header("CV Scanner MVP")

# Initialize global embeddings model
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Preinitialize the LLM
if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name='gpt-3.5-turbo')

# This chain coordinates the LLM for the specific QA task
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = load_qa_chain(st.session_state.llm, chain_type="stuff")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your CV", type="pdf", on_change=st.cache_resource.clear)

@st.cache_resource
def create_document(pdf):
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)

    # Filter content: remove blank lines
    text = "\n".join(line for line in text.split("\n") if line.strip())

    # Create a Document instance with the full text
    document = Document(page_content=text, metadata={})
    return [document]

# If a PDF file is uploaded
if pdf_file:
    documents = create_document(pdf_file)
    job_description = st.text_area("Provide the job description:")

    if job_description:
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

        match_prompt_template = f"""
            Based on the provided CV information: {documents}
            and the job position details: {job_description},

            Please perform the following tasks:

            1. **List the Candidate's Main Skills and Experiences**:
            - Identify and enumerate the primary skills and experiences mentioned in the CV.

            2. **Match Skills to Job Position**:
            - Provide a list of skills from the candidate's CV that directly fit the job position.
            - If there are no relevant skills, indicate with a single bullet:
                * There are no skills that fit the job position.

            4. **Match Score: 0-100%**
            - Provide a match score from 0% to 100% indicating how well the candidate fits the job position.
            - Use 0% if the job position has nothing to do with the candidate's skills or is in a completely different field.
            - Justify the match score by highlighting key similarities or discrepancies between the candidate's qualifications and the job requirements.
        """

        answer = st.session_state.qa_chain.invoke(input={"input_documents": documents, "question": match_prompt_template})
        st.write(answer['output_text'])
